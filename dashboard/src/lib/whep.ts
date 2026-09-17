/**
 * Minimal WHEP (WebRTC-HTTP Egress Protocol) client for MediaMTX's WebRTC
 * output — the low-latency path for the live wall.
 *
 * Why this exists: HLS has a latency floor. Even in low-latency mode with 1s
 * segments and the player pinned to the live edge (see CameraTile's hls.js
 * tuning), it lands seconds behind the camera. That costs twice over on a
 * surveillance wall — an operator watches the recent past, and the detection
 * overlay desynchronises, because its boxes arrive over a near-instant
 * WebSocket and therefore run *ahead* of the picture they are drawn on
 * (DetectionOverlay.tsx documented this as a known limitation). WebRTC brings
 * the video to roughly a few hundred milliseconds, which both fixes the lag
 * and puts the boxes back on top of the objects they describe.
 *
 * Non-trickle on purpose: ICE gathers fully before the single POST, so there
 * is no PATCH/Location round-trip to manage. This deployment is LAN- and
 * localhost-shaped, where host candidates appear immediately, so waiting costs
 * nothing and the whole handshake stays one request.
 */

/** Give up gathering and offer whatever we have. A STUN server that never
 * answers must not hold a camera tile hostage; on a LAN the host candidates
 * that matter are already in hand well before this fires. */
const ICE_GATHER_TIMEOUT_MS = 1500;

export class WhepSession {
  private pc: RTCPeerConnection | null = null;
  private resource: string | null = null;
  private closed = false;

  /**
   * Negotiate a receive-only WebRTC session and attach it to `video`.
   *
   * Resolves once the remote description is set — not once media flows, which
   * the caller should confirm from the <video> element itself, the same way
   * the HLS path does. Rejects if the handshake fails, which is the caller's
   * cue to fall back to HLS.
   *
   * `onFailed` covers the *later* failure: a session that negotiated fine and
   * then dropped (camera unplugged, network partition). Without it a dead
   * tile would sit on a frozen last frame indefinitely.
   */
  async start(url: string, video: HTMLVideoElement, onFailed?: () => void): Promise<void> {
    const pc = new RTCPeerConnection({ iceServers: [] });
    this.pc = pc;

    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    const stream = new MediaStream();
    pc.ontrack = (event) => {
      stream.addTrack(event.track);
      if (video.srcObject !== stream) video.srcObject = stream;
    };
    pc.addEventListener('connectionstatechange', () => {
      if (this.closed) return;
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        onFailed?.();
      }
    });

    await pc.setLocalDescription(await pc.createOffer());
    await this.gatherIce(pc);
    if (this.closed) return;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/sdp' },
      body: pc.localDescription?.sdp ?? '',
    });
    if (!response.ok) {
      throw new Error(`whep refused the offer (${response.status})`);
    }
    const answer = await response.text();
    // MediaMTX answers with a RELATIVE Location ("/<path>/whep/<session>").
    // Resolving it against `url` rather than using it raw matters: the
    // dashboard is served from a different origin than MediaMTX, so a bare
    // fetch() of that path would send the DELETE to the API instead of the
    // media server and leak the session. (MediaMTX does list Location in
    // Access-Control-Expose-Headers, so this is readable cross-origin.)
    const location = response.headers.get('location');
    this.resource = location ? new URL(location, url).toString() : null;
    if (this.closed) return;

    await pc.setRemoteDescription({ type: 'answer', sdp: answer });
  }

  close(): void {
    this.closed = true;
    if (this.resource) {
      // Fire-and-forget: the tile is already gone by the time this lands.
      void fetch(this.resource, { method: 'DELETE' }).catch(() => undefined);
      this.resource = null;
    }
    this.pc?.close();
    this.pc = null;
  }

  private gatherIce(pc: RTCPeerConnection): Promise<void> {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const finish = () => {
        window.clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', onChange);
        resolve();
      };
      const onChange = () => {
        if (pc.iceGatheringState === 'complete') finish();
      };
      const timer = window.setTimeout(finish, ICE_GATHER_TIMEOUT_MS);
      pc.addEventListener('icegatheringstatechange', onChange);
    });
  }
}
