import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import type { Camera, StreamInfo } from '@/types';
import { api } from '@/lib/api';
import { StatusDot } from './Primitives';
import { DetectionOverlay } from './DetectionOverlay';

/**
 * One camera on the live wall.
 *
 * Transport order is WebRTC (WHEP) first, HLS second, and that order is the
 * point rather than a preference. MediaMTX's WebRTC path delivers in a few
 * hundred milliseconds; its low-latency HLS still lands seconds behind,
 * because HLS is segments over HTTP and a player has to buffer some of them
 * before it will start. On an operator's wall those seconds decide whether
 * somebody is watching an incident or reading about one, and they also decide
 * whether DetectionOverlay's boxes sit on the object or lead it (see that
 * file's note on time). RTSP is never an option here: no browser plays it.
 *
 * HLS stays as the fallback, not as dead code -- WebRTC needs a reachable UDP
 * media port (infra/mediamtx.yml's webrtcLocalUDPAddress, published in
 * docker-compose.yml), and a network that blocks it must degrade to a
 * late picture rather than to no picture.
 */
export function CameraTile({
  camera,
  onSelect,
  onConnect,
  onDisconnect,
}: {
  camera: Camera;
  onSelect?: () => void;
  /** Slot has no live source bound yet — clicking it should ask for an IP
   * instead of trying (and failing) to open a stream. */
  onConnect?: () => void;
  /** Bound camera — tear it back down to an empty slot. */
  onDisconnect?: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<'connecting' | 'live' | 'down'>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [transport, setTransport] = useState<'webrtc' | 'hls' | null>(null);

  useEffect(() => {
    if (!camera.enabled) return undefined;

    let cancelled = false;
    let hls: Hls | null = null;
    let peer: RTCPeerConnection | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    let graceTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let everLive = false;
    // Kept so the WebRTC watchdog can hand the same stream URLs to HLS
    // without a second round trip to the API.
    let lastInfo: StreamInfo | null = null;

    // A camera that was just bound by ConnectCameraModal needs real grace:
    // MediaMTX accepts the source the moment it is configured, but still has
    // to finish the RTSP handshake with the camera, so the first attempts
    // legitimately find nothing published yet.
    //
    // Once a tile HAS been live, the same patience becomes a lie -- a stopped
    // camera would sit on a frozen frame for fifteen seconds before admitting
    // it. So the budget collapses as soon as we have seen real video.
    const FIRST_CONNECT_ATTEMPTS = 6;
    const RECONNECT_ATTEMPTS = 2;
    const RETRY_DELAY_MS = 2000;
    // Negotiation succeeding while ICE never connects means the media port is
    // unreachable. Do not sit on a blank tile waiting: fall back to HLS.
    const WEBRTC_ICE_TIMEOUT_MS = 5000;
    // WebRTC reports a brief 'disconnected' on ordinary packet loss and then
    // recovers. Only a drop that outlasts this is worth reporting.
    const WEBRTC_DROP_GRACE_MS = 3000;

    function clearTimers() {
      if (watchdog) clearTimeout(watchdog);
      if (graceTimer) clearTimeout(graceTimer);
      watchdog = undefined;
      graceTimer = undefined;
    }

    function teardownPeer() {
      clearTimers();
      if (peer) {
        peer.onconnectionstatechange = null;
        peer.ontrack = null;
        peer.close();
        peer = null;
      }
      const video = videoRef.current;
      if (video && video.srcObject) video.srcObject = null;
    }

    function teardownHls() {
      hls?.destroy();
      hls = null;
    }

    function goLive(kind: 'webrtc' | 'hls') {
      if (cancelled) return;
      clearTimers();
      everLive = true;
      attempt = 0;
      setTransport(kind);
      setError(null);
      setState('live');
    }

    function fail(message: string) {
      if (cancelled) return;
      teardownPeer();
      teardownHls();
      attempt += 1;
      const budget = everLive ? RECONNECT_ATTEMPTS : FIRST_CONNECT_ATTEMPTS;
      if (attempt < budget) {
        setState('connecting');
        retryTimer = setTimeout(connect, RETRY_DELAY_MS);
      } else {
        setState('down');
        setError(message);
      }
    }

    /** Wait for ICE gathering, with a ceiling. MediaMTX is on the LAN, so host
     * candidates arrive immediately; a candidate that never comes must not
     * hang the tile forever. */
    function waitForIce(pc: RTCPeerConnection): Promise<void> {
      if (pc.iceGatheringState === 'complete') return Promise.resolve();
      return new Promise((resolve) => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const finish = () => {
          pc.removeEventListener('icegatheringstatechange', check);
          if (timer) clearTimeout(timer);
          resolve();
        };
        const check = () => {
          if (pc.iceGatheringState === 'complete') finish();
        };
        pc.addEventListener('icegatheringstatechange', check);
        timer = setTimeout(finish, 1200);
      });
    }

    /** WHEP: POST an SDP offer, get an answer back. Resolves true once the
     * exchange succeeded — going 'live' is then the connection state's job. */
    async function tryWebRtc(whepUrl: string, video: HTMLVideoElement): Promise<boolean> {
      if (typeof RTCPeerConnection === 'undefined') return false;

      const pc = new RTCPeerConnection({ iceServers: [] });
      peer = pc;
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.addTransceiver('audio', { direction: 'recvonly' });

      pc.ontrack = (event) => {
        if (cancelled || peer !== pc) return;
        const [stream] = event.streams;
        if (stream) {
          video.srcObject = stream;
          // autoPlay + muted should cover this; a rejected play() is not fatal.
          void video.play().catch(() => undefined);
        }
      };

      pc.onconnectionstatechange = () => {
        if (cancelled || peer !== pc) return;
        switch (pc.connectionState) {
          case 'connected':
            if (graceTimer) {
              clearTimeout(graceTimer);
              graceTimer = undefined;
            }
            goLive('webrtc');
            break;
          case 'disconnected':
            if (!graceTimer) {
              graceTimer = setTimeout(() => {
                if (!cancelled && peer === pc) fail('signal lost');
              }, WEBRTC_DROP_GRACE_MS);
            }
            break;
          case 'failed':
          case 'closed':
            fail('signal lost');
            break;
          default:
            break;
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIce(pc);
      if (cancelled || peer !== pc) return false;

      const response = await fetch(whepUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: pc.localDescription?.sdp ?? offer.sdp ?? '',
      });
      if (!response.ok) throw new Error(`whep ${response.status}`);
      const answer = await response.text();
      if (cancelled || peer !== pc) return false;
      await pc.setRemoteDescription({ type: 'answer', sdp: answer });

      watchdog = setTimeout(() => {
        if (cancelled || peer !== pc || pc.connectionState === 'connected') return;
        // Negotiated but never connected: the media port is not reachable.
        teardownPeer();
        const info = lastInfo;
        if (info && videoRef.current) startHls(info, videoRef.current);
      }, WEBRTC_ICE_TIMEOUT_MS);

      return true;
    }

    function startHls(info: StreamInfo, video: HTMLVideoElement) {
      if (cancelled) return;
      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS (Safari). Wait for the browser to confirm playable media
        // before calling this 'live' -- setting .src succeeds even when
        // nothing is published at that MediaMTX path, and a green dot over a
        // blank tile teaches an operator to trust a dead camera.
        video.src = info.hls_url;
        video.addEventListener('loadedmetadata', () => goLive('hls'), { once: true });
        video.addEventListener('error', () => fail('stream unavailable'), { once: true });
        return;
      }
      if (!Hls.isSupported()) {
        setState('down');
        setError('no WebRTC or HLS support in this browser');
        return;
      }
      hls = new Hls({
        lowLatencyMode: true,
        backBufferLength: 10,
        maxBufferLength: 6,
        // Defaults sit ~3 segments behind the live edge. That is a sensible
        // VOD margin and far too much for a security feed, so hug the edge
        // and catch up fast rather than settling into a standing delay.
        liveSyncDurationCount: 1,
        liveMaxLatencyDurationCount: 3,
        maxLiveSyncPlaybackRate: 5,
      });
      hls.loadSource(info.hls_url);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => goLive('hls'));
      hls.on(Hls.Events.ERROR, (_e, data) => {
        if (data.fatal) fail(data.details);
      });
    }

    function connect() {
      if (cancelled) return;
      teardownPeer();
      teardownHls();

      void (async () => {
        try {
          const info = await api.stream(camera.id);
          if (cancelled || !videoRef.current) return;
          lastInfo = info;
          const video = videoRef.current;

          try {
            if (await tryWebRtc(info.webrtc_url, video)) return;
          } catch {
            // WebRTC unavailable or refused — that is what HLS is here for.
            teardownPeer();
          }
          if (cancelled) return;
          startHls(info, video);
        } catch (err) {
          fail(err instanceof Error ? err.message : 'stream unavailable');
        }
      })();
    }

    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      teardownPeer();
      teardownHls();
    };
  }, [camera.id, camera.enabled]);

  if (!camera.enabled) {
    return (
      <button
        type="button"
        onClick={onConnect}
        className="group relative flex aspect-video flex-col items-center justify-center gap-2
                   border border-dashed border-rule2 bg-panel/40 text-dim transition-colors
                   hover:border-signal hover:text-signal"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-7 w-7">
          <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.6l1-1.6A1.5 1.5 0 0 1 9.4 4.7h5.2a1.5 1.5 0 0 1 1.3.7l1 1.6h2.6A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5v-9Z" />
          <circle cx="12" cy="13" r="3.6" />
          <path d="M12 11v4M10 13h4" />
        </svg>
        <span className="font-mono text-2xs uppercase tracking-[0.16em]">Connect camera</span>
        <span className="font-mono text-2xs text-dim/70">{camera.code}</span>
      </button>
    );
  }

  return (
    <div
      role={onSelect ? 'button' : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onClick={onSelect}
      className="panel group relative aspect-video overflow-hidden text-left transition-colors hover:border-signal/60"
    >
      {onDisconnect && (
        <button
          type="button"
          title="Disconnect camera"
          onClick={(e) => {
            e.stopPropagation();
            onDisconnect();
          }}
          className="absolute right-2 top-2 z-10 flex h-6 w-6 items-center justify-center
                     border border-rule2 bg-void/80 font-mono text-2xs text-dim opacity-0
                     transition-opacity hover:border-alarm hover:text-alarm group-hover:opacity-100"
        >
          ✕
        </button>
      )}

      <video
        ref={videoRef}
        muted
        playsInline
        autoPlay
        className="h-full w-full bg-void object-cover opacity-90 transition-opacity group-hover:opacity-100"
      />

      <DetectionOverlay active={state === 'live'} camera={camera} />

      {state !== 'live' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-void/85">
          <span className="font-mono text-2xs uppercase tracking-[0.2em] text-dim">
            {state === 'connecting' ? 'acquiring signal' : 'no signal'}
          </span>
          {error && <span className="px-4 text-center font-mono text-2xs text-alarm">{error}</span>}
        </div>
      )}

      {/* Overlay bar. Kept to one line so it never eats the picture. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-void via-void/80 to-transparent px-2.5 py-1.5">
        <span className="flex items-center gap-2">
          <StatusDot state={state === 'live' ? 'live' : state === 'down' ? 'down' : 'warn'} />
          <span className="font-display text-xs font-semibold tracking-[0.1em] text-bright">
            {camera.code}
          </span>
          <span className="truncate font-mono text-2xs text-dim">{camera.name}</span>
        </span>
        <span className="flex items-center gap-2 font-mono text-2xs tabular-nums text-dim">
          {/* Which transport won matters when a feed looks late: RTC is
              sub-second, HLS is the seconds-behind fallback. */}
          {state === 'live' && transport && (
            <span className={transport === 'webrtc' ? 'text-phosphor' : 'text-ember'}>
              {transport === 'webrtc' ? 'RTC' : 'HLS'}
            </span>
          )}
          <span>{camera.analytics_fps.toFixed(0)} fps</span>
        </span>
      </div>
    </div>
  );
}
