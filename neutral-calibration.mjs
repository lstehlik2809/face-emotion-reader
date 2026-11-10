export const NEUTRAL_STORE_KEY = "facereader.neutralCalibration.v1";

// Keep the versioned vector in a fixed order. Blinks, gaze, and _neutral are
// deliberately omitted so looking at the screen and blinking don't count as expressions.
const CHANNELS = [
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight",
  "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
  "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
  "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
  "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
  "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
  "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
];
const CHANNEL_INDEX = new Map(CHANNELS.map((name, i) => [name, i]));
const FACING_MODES = ["user", "environment", "left", "right"];

// Heuristic thresholds. Neutral bias fades out with head angle or a change in
// any expression channel, leaving the ordinary scores at either cutoff.
const FULL_POSE_DEGREES = 7;
const MAX_POSE_DEGREES = 15;
const FULL_EXPRESSION_DELTA = 0.03;
const MAX_EXPRESSION_DELTA = 0.12;
const MAX_NEUTRAL_WEIGHT = 0.65;

export function cameraKey(settings, requestedFacing) {
  const id = settings?.deviceId;
  if (typeof id === "string" && id) {
    // FNV-1a on the browser's opaque device ID. Keep the fingerprint numeric
    // and separate from the facing-mode fallback codes.
    let hash = 2166136261;
    for (let i = 0; i < id.length; i++) hash = Math.imul(hash ^ id.charCodeAt(i), 16777619);
    return -(hash >>> 0) - 1;
  }
  const reported = FACING_MODES.indexOf(settings?.facingMode);
  const requested = FACING_MODES.indexOf(requestedFacing);
  return reported >= 0 ? reported : Math.max(0, requested);
}

function validExpression(expression) {
  return Array.isArray(expression) && expression.length === CHANNELS.length &&
    expression.every(value => Number.isFinite(value) && value >= 0 && value <= 1);
}

export function parseNeutralProfile(raw) {
  try {
    const { camera, forward, right, expression } = JSON.parse(raw);
    if (!Number.isInteger(camera) || camera < -4294967296 || camera >= FACING_MODES.length ||
        !isUnitVector(forward) || !isUnitVector(right) || Math.abs(dot(forward, right)) > 0.01 ||
        !validExpression(expression)) return null;
    return { camera, ...orthonormalPose(forward, right), expression };
  } catch (_) { return null; }
}

function isUnitVector(vector) {
  return Array.isArray(vector) && vector.length === 3 && vector.every(Number.isFinite) &&
    Math.abs(Math.hypot(...vector) - 1) <= 0.01;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function unitVector(vector) {
  const length = Math.hypot(...vector);
  return Number.isFinite(length) && length > 1e-6 ? vector.map(value => value / length) : null;
}

function orthonormalPose(forward, right) {
  const z = unitVector(forward);
  if (!z) return null;
  const projection = dot(z, right);
  const x = unitVector(right.map((value, i) => value - projection * z[i]));
  return x ? { forward: z, right: x } : null;
}

export function faceOrientation(matrix) {
  const data = matrix?.data;
  if (matrix?.rows !== 4 || matrix.columns !== 4 || data?.length !== 16 ||
      !Array.from(data).every(Number.isFinite) || data[3] !== 0 || data[7] !== 0 ||
      data[11] !== 0 || data[15] !== 1) return null;
  // MediaPipe Tasks sends MatrixData in column-major order. The third and first
  // columns transform canonical +Z and +X, capturing forward direction and roll.
  return orthonormalPose([data[8], data[9], data[10]], [data[0], data[1], data[2]]);
}

export function neutralSample(matrix, categories) {
  const orientation = faceOrientation(matrix);
  if (!orientation || !Array.isArray(categories)) return null;
  const expression = new Array(CHANNELS.length).fill(NaN);
  for (const category of categories) {
    const i = CHANNEL_INDEX.get(category?.categoryName);
    if (i !== undefined) expression[i] = category.score;
  }
  return validExpression(expression) ? { ...orientation, expression } : null;
}

function poseDegrees(a, b) {
  const upA = cross(a.forward, a.right), upB = cross(b.forward, b.right);
  // The trace of the relative rotation is the sum of the corresponding axis
  // dot products. It equals 1 + 2 cos(angle), including roll about face forward.
  const trace = dot(a.forward, b.forward) + dot(a.right, b.right) + dot(upA, upB);
  const cosine = (trace - 1) / 2;
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

function expressionDelta(a, b) {
  return Math.max(...a.map((value, i) => Math.abs(value - b[i])));
}

export function stableNeutralSample(reference, sample) {
  return poseDegrees(reference, sample) <= FULL_POSE_DEGREES &&
    expressionDelta(reference.expression, sample.expression) <= MAX_EXPRESSION_DELTA;
}

function median(values) {
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

export function makeNeutralProfile(camera, samples) {
  const forward = [0, 1, 2].map(i => median(samples.map(sample => sample.forward[i])));
  const right = [0, 1, 2].map(i => median(samples.map(sample => sample.right[i])));
  return {
    camera,
    ...orthonormalPose(forward, right),
    expression: CHANNELS.map((_, i) => median(samples.map(sample => sample.expression[i])))
  };
}

function fade(value, full, cutoff) {
  return Math.max(0, Math.min(1, (cutoff - value) / (cutoff - full)));
}

export function neutralAdjustment(profile, camera, sample) {
  if (!profile) return { weight: 0 };
  if (profile.camera !== camera) return { weight: 0, cue: "camera" };
  if (!sample) return { weight: 0 };
  const pose = fade(poseDegrees(profile, sample), FULL_POSE_DEGREES, MAX_POSE_DEGREES);
  if (!pose) return { weight: 0, cue: "pose" };
  const expression = fade(expressionDelta(profile.expression, sample.expression),
    FULL_EXPRESSION_DELTA, MAX_EXPRESSION_DELTA);
  return { weight: MAX_NEUTRAL_WEIGHT * pose * expression };
}

export function blendNeutral(probs, weight) {
  if (!weight) return probs;
  const adjusted = {};
  for (const key of Object.keys(probs)) adjusted[key] = probs[key] * (1 - weight);
  adjusted.neutral += weight;
  return adjusted;
}
