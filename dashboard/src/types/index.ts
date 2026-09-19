/** Mirrors the API schemas in api/src/ibvap_api/schemas.py (BUILD_SPEC §8, §9). */

export type Severity = 'info' | 'low' | 'medium' | 'high' | 'critical';
export type AlertStatus = 'raised' | 'acknowledged' | 'adjudicated';
export type Adjudication = 'true_positive' | 'false_positive' | 'unclear';
export type Role = 'viewer' | 'operator' | 'investigator' | 'admin';
export type StreamState = 'connecting' | 'live' | 'stalled' | 'reconnecting' | 'failed';
export type ZoneKind = 'area' | 'tripwire' | 'mask';

export interface Site {
  id: string;
  code: string;
  name: string;
  sector: string | null;
  timezone: string;
}

export interface Camera {
  id: string;
  site_id: string;
  code: string;
  name: string;
  mediamtx_path: string;
  resolution_w: number;
  resolution_h: number;
  analytics_fps: number;
  is_recording_only: boolean;
  enabled: boolean;
}

export interface StreamInfo {
  camera_id: string;
  mediamtx_path: string;
  hls_url: string;
  webrtc_url: string;
}

export interface Zone {
  id: string;
  camera_id: string;
  name: string;
  kind: ZoneKind;
  polygon: number[][];
  direction: string | null;
  classes: string[] | null;
  severity_base: number;
  enabled: boolean;
}

/** One contribution to a risk score. `weight` may be negative (P3). */
export interface RiskContribution {
  code: string;
  weight: number;
  detail: Record<string, unknown>;
}

export interface AlertSummary {
  id: string;
  site_id: string;
  camera_id: string;
  kind: string;
  severity: Severity;
  risk_score: number;
  reason_codes: string[];
  ts_utc: string;
  status: AlertStatus;
  adjudication: Adjudication | null;
  ledger_status: string;
}

export interface EvidenceItem {
  id: string;
  kind: string;
  sha256: string;
  bytes: number | null;
  width: number | null;
  height: number | null;
  captured_at: string | null;
  enhanced: boolean;
  enhancement_params: Record<string, unknown> | null;
}

export interface AlertDetail extends AlertSummary {
  track_id: number | null;
  zone_id: string | null;
  window_start: string | null;
  window_end: string | null;
  risk_breakdown: RiskContribution[];
  evidence_hash: string;
  evidence_doc: Record<string, unknown>;
  items: EvidenceItem[];
}

export interface AlertPage {
  items: AlertSummary[];
  next_cursor: string | null;
}

export interface VerificationCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface MerkleProof {
  leaf_index: number;
  leaf_hash: string;
  proof: [string, string][];
  computed_root: string;
  stored_root: string;
  root_match: boolean;
}

export interface LedgerInfo {
  backend: string;
  tx_id: string | null;
  block_number: number | null;
  anchored_at: string | null;
  root_on_chain: string | null;
  chain_match: boolean;
}

export type Verdict = 'VERIFIED' | 'PENDING_ANCHOR' | 'TAMPERED' | 'UNVERIFIABLE';

export interface Verification {
  alert_id: string;
  stored_hash: string;
  recomputed_hash: string;
  hash_match: boolean;
  canonical_bytes_sha256: string;
  canonical_length: number;
  evidence_items: { kind: string; object_key: string; sha256: string }[];
  merkle: MerkleProof | null;
  ledger: LedgerInfo | null;
  verdict: Verdict;
  checks: VerificationCheck[];
  diff: string[];
}

export interface ComponentHealth {
  name: string;
  ok: boolean;
  detail: string;
  latency_ms: number | null;
}

export interface Health {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  components: ComponentHealth[];
  warnings: string[];
}

export interface Session {
  access_token: string;
  refresh_token: string;
  role: Role;
  display_name: string;
}

/** Server → client WebSocket envelope (§9). */
export type ServerMsg =
  | { type: 'hello'; server_time: string; subscribed: string[] }
  | { type: 'alert'; alert: AlertSummary }
  | { type: 'alert_update'; alert_id: string; status: AlertStatus }
  | { type: 'stream_health'; camera_id: string; state: StreamState; fps: number }
  | { type: 'evqm'; camera_id: string; profile: string }
  | { type: 'heartbeat'; ts: string };

/** One real tracked object, as published live by the worker (pipeline.py's
 * `_on_tracks` hook) -- box is in the camera's ORIGINAL pixel coordinates. */
/** Present on a track only while it has a currently-held weapon reading --
 * the worker publishes this straight from its own held-state cache
 * (pipeline.py's `_weapon_last`), so it keeps appearing on a motionless
 * track for the same reason the alert itself keeps firing: stillness must
 * never read as "put it down." `box` is already in this frame's original
 * pixel space, mapped back from the crop the model actually saw -- draw it
 * with the exact same transform used for the track's own `box` below. */
export interface LiveWeaponHit {
  track_id: number;
  weapon_type: string;
  conf: number;
  frames_agreed: number;
  frames_seen: number;
  box: [number, number, number, number] | null;
}

export interface LiveTrack {
  track_id: number;
  cls: string;
  box: [number, number, number, number];
  speed_px_s: number;
  weapon: LiveWeaponHit | null;
}

/** `/ws/live/{camera_id}` message: one frame's worth of real tracks.
 *
 * `frame_w`/`frame_h` are the true pixel size of the frame the boxes were
 * measured in. Always prefer them over the camera row's `resolution_w/h`,
 * which is a static seed-time guess and wrong for any camera that negotiated
 * a different size. Optional only because an older worker may not send them. */
export interface LiveTrackFrame {
  type: 'tracks';
  camera_id: string;
  ts: string;
  tracks: LiveTrack[];
  frame_w?: number;
  frame_h?: number;
}
