import assert from "node:assert/strict";
import { test } from "node:test";
import {
  blendNeutral, cameraKey, faceOrientation, makeNeutralProfile, neutralAdjustment,
  neutralSample, parseNeutralProfile, stableNeutralSample
} from "../neutral-calibration.mjs";

// MediaPipe Tasks Face Landmarker v0.10.35 category names.
const categories = `
  _neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
  cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft
  eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft eyeLookOutRight eyeLookUpLeft
  eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight jawForward jawLeft
  jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight
  mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight
  mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper
  mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight mouthUpperUpLeft
  mouthUpperUpRight noseSneerLeft noseSneerRight
`.trim().split(/\s+/);

function pose(degrees = 0, scale = 1) {
  const a = degrees * Math.PI / 180;
  const c = Math.cos(a) * scale, s = Math.sin(a) * scale;
  return { rows: 4, columns: 4, data: [c, 0, -s, 0, 0, scale, 0, 0, s, 0, c, 0, -1, 22, -65, 1] };
}

function readings(changes = {}) {
  return categories.map(categoryName => ({
    categoryName, score: changes[categoryName] ?? 0.21
  })).reverse();
}

function sample(degrees = 0, changes = {}) {
  return neutralSample(pose(degrees), readings(changes));
}

const frontCamera = cameraKey({ deviceId: "front" }, "user");
const base = { angry: 0.55, neutral: 0.25, happy: 0.08, sad: 0.06,
  fearful: 0.02, contempt: 0.01, surprised: 0.02, disgusted: 0.01 };

test("reads orientation from the packed first and third columns, removing translation and scale", () => {
  const matrix = { rows: 4, columns: 4, data: new Float32Array([
    0, 0, 2, 0, -2, 0, 0, 0, 0, -2, 0, 0, 1, 22, -65, 1
  ]) };
  assert.deepEqual(faceOrientation(matrix), { forward: [0, -1, 0], right: [0, 0, 1] });
  assert.deepEqual(faceOrientation(pose(45)), faceOrientation(pose(45, 2)));
  assert.equal(faceOrientation(pose(0, 0)), null);
  assert.equal(faceOrientation({ ...matrix, rows: 3 }), null);
  const rowMajor = matrix.data.map((_, i) => matrix.data[(i % 4) * 4 + Math.floor(i / 4)]);
  assert.equal(faceOrientation({ ...matrix, data: rowMajor }), null);
  assert.equal(faceOrientation(undefined), null);
});

test("builds a median baseline from named channels and rejects incomplete readings", () => {
  const middle = sample(0, { browDownLeft: 0.24 });
  const higher = sample(0, { browDownLeft: 0.31 });
  const lower = sample(0, { browDownLeft: 0.22 });
  const profile = makeNeutralProfile(frontCamera, [higher, lower, middle]);
  assert.deepEqual(profile.expression, middle.expression);
  assert.deepEqual(profile.forward, [0, 0, 1]);
  const expectedRight = [1, 0, 0];
  assert.ok(profile.right.every((value, i) => value === expectedRight[i]));
  assert.equal(neutralSample(pose(), [{ categoryName: "browDownLeft", score: 0.2 }]), null);
  assert.equal(sample(0, { eyeSquintRight: Infinity }), null);
  assert.deepEqual(sample(0, { eyeBlinkLeft: 0.95, eyeLookDownRight: 0.9, _neutral: 0.8 }), sample());
  assert.equal(stableNeutralSample(sample(), sample(18)), false);
  assert.equal(stableNeutralSample(sample(), sample(0, { browDownLeft: 0.6 })), false);
});

test("neutral calibration lowers false Angry and preserves the probability total", () => {
  const profile = makeNeutralProfile(frontCamera, [sample()]);
  const { weight } = neutralAdjustment(profile, frontCamera, sample());
  const adjusted = blendNeutral(base, weight);
  assert.ok(adjusted.angry < base.angry);
  assert.ok(Object.entries(adjusted).every(([key, score]) => key === "neutral" || score < adjusted.neutral));
  assert.equal(base.angry, 0.55);
  assert.deepEqual(Object.keys(adjusted), Object.keys(base));
  assert.ok(Math.abs(Object.values(adjusted).reduce((a, b) => a + b) - 1) < 1e-12);
  const intermediate = neutralAdjustment(profile, frontCamera, sample(11, { browDownLeft: 0.26 }));
  assert.ok(intermediate.weight > 0 && intermediate.weight < weight);
});

test("ordinary scores return for missing data, meaningful expression changes or a different pose or camera", () => {
  const profile = makeNeutralProfile(frontCamera, [sample()]);
  const changedBrow = sample(0, { browDownLeft: 0.39 });
  const turnedHead = sample(24);
  const rearCamera = cameraKey({ deviceId: "rear" }, "environment");
  for (const [reference, camera, frame, expectedCue] of [
    [null, frontCamera, sample(), undefined],
    [profile, frontCamera, null, undefined],
    [profile, frontCamera, changedBrow, undefined],
    [profile, frontCamera, turnedHead, "pose"],
    [profile, rearCamera, sample(), "camera"]
  ]) {
    const { weight, cue } = neutralAdjustment(reference, camera, frame);
    assert.equal(cue, expectedCue);
    assert.equal(blendNeutral(base, weight), base);
  }
  assert.ok(neutralAdjustment(profile, frontCamera, sample()).weight > 0);
});

test("rolling sideways without changing face forward restores ordinary scores and restarts the hold", () => {
  const original = sample();
  const profile = makeNeutralProfile(frontCamera, [original]);
  const matrix = pose();
  const cos30 = Math.sqrt(3) / 2;
  matrix.data.splice(0, 8, cos30, 0.5, 0, 0, -0.5, cos30, 0, 0);
  const rolled = neutralSample(matrix, readings());
  assert.deepEqual(rolled.forward, original.forward);
  assert.deepEqual(rolled.expression, original.expression);
  assert.equal(stableNeutralSample(original, rolled), false);
  const { weight, cue } = neutralAdjustment(profile, frontCamera, rolled);
  assert.equal(cue, "pose");
  assert.equal(blendNeutral(base, weight), base);
  assert.ok(neutralAdjustment(profile, frontCamera, original).weight > 0);
  const rolledProfile = makeNeutralProfile(frontCamera, [rolled]);
  assert.ok(neutralAdjustment(rolledProfile, frontCamera, rolled).weight > 0);
});

test("reload validates the numeric reference and camera identification falls back to facing", () => {
  const profile = makeNeutralProfile(frontCamera, [sample()]);
  const raw = JSON.stringify(profile);
  assert.deepEqual(parseNeutralProfile(raw), JSON.parse(raw));
  for (const invalid of [null, "invalid", "{}", JSON.stringify({ ...profile, camera: "front" }),
    JSON.stringify({ ...profile, forward: [0, 0, 0] }), JSON.stringify({ ...profile, right: undefined }),
    JSON.stringify({ ...profile, right: [0, 0, 1] }), JSON.stringify({ ...profile, expression: [0.2] }),
    JSON.stringify({ ...profile, expression: profile.expression.map(() => 1.01) })]) {
    assert.equal(parseNeutralProfile(invalid), null);
  }
  assert.equal(cameraKey({ deviceId: "front" }, "environment"), frontCamera);
  assert.notEqual(cameraKey({ deviceId: "other" }, "user"), frontCamera);
  assert.equal(cameraKey({ facingMode: "environment" }, "user"), cameraKey({}, "environment"));
  assert.notEqual(cameraKey({}, "environment"), cameraKey({}, "user"));
});
