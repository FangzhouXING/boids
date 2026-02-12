const canvas = document.getElementById('boidCanvas');
const restartButton = document.getElementById('restartButton');
const turnAccelRange = document.getElementById('turnAccelRange');
const turnAccelValue = document.getElementById('turnAccelValue');
const minSpeedRange = document.getElementById('minSpeedRange');
const minSpeedValue = document.getElementById('minSpeedValue');
const predatorAttentionRange = document.getElementById('predatorAttentionRange');
const predatorAttentionValue = document.getElementById('predatorAttentionValue');
const viewToggleButton = document.getElementById('viewToggleButton');

const FIXED_STEP = 1 / 60;
const BOID_COUNT = 10000;
const PREDATOR_COUNT = 5;
const BOID_WORKGROUP_SIZE = 128;
const CATCH_CLEAR_WORKGROUP_SIZE = 32;
const BOID_FLOATS = 8;
const PREDATOR_FLOATS = 12;

const config = {
  perceptionRadius: 72,
  separationRadius: 22,
  predatorAvoidRadius: 125,
  maxSpeed: 1.7,
  minSpeed: Number(minSpeedRange.value),
  maxForce: 0.04,
  maxTurnRate: (220 * Math.PI) / 180,
  maxTurnAcceleration: (Number(turnAccelRange.value) * Math.PI) / 180,
  alignWeight: 0.85,
  cohesionWeight: 0.65,
  separationWeight: 1.35,
  predatorAvoidWeight: 2.4,
  predatorCatchRadius: 9,
  predatorSeparationRadius: 50,
  predatorSeparationWeight: 1.8,
  predatorTurnRateFactor: 0.85,
  predatorTurnAccelerationFactor: 0.8,
  predatorSpeedFactor: 1.03,
  predatorPostCatchPauseMinSeconds: 0,
  predatorPostCatchPauseMaxSeconds: 0,
  predatorPauseSlowdownSeconds: 0,
  predatorAttentionSeconds: Number(predatorAttentionRange.value),
};

let worldWidth = 960;
let worldHeight = 560;
let devicePixelRatioCached = Math.max(window.devicePixelRatio || 1, 1);
let frameIndex = 0;
let accumulator = 0;
let lastFrameTime = 0;
let currentBoidBufferIndex = 0;
let renderMode = 'boids';
let gpu = null;
let gpuRuntimeError = false;
let lastTextState = JSON.stringify({
  mode: 'initializing',
  boidCount: BOID_COUNT,
  predatorCount: PREDATOR_COUNT,
});

const boidUpdateShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct PredatorState {
  posVel: vec4f,
  headingTimers: vec4f,
  aux: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

@group(0) @binding(0) var<storage, read> boidsIn: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> boidsOut: array<BoidState>;
@group(0) @binding(2) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(3) var<storage, read_write> caughtFlags: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

const TAU: f32 = 6.28318530718;

fn wrap_coordinate(value: f32, size: f32) -> f32 {
  var result = value;
  if (result < 0.0) {
    result = result + size;
  }
  if (result >= size) {
    result = result - size;
  }
  return result;
}

fn wrapped_delta(srcValue: f32, dstValue: f32, size: f32) -> f32 {
  var delta = dstValue - srcValue;
  let half = size * 0.5;
  if (delta > half) {
    delta = delta - size;
  } else if (delta < -half) {
    delta = delta + size;
  }
  return delta;
}

fn limit_magnitude(v: vec2f, maxLen: f32) -> vec2f {
  let len = length(v);
  if (len <= maxLen || len <= 0.000001) {
    return v;
  }
  return v * (maxLen / len);
}

fn normalize_angle(angle: f32) -> f32 {
  var a = angle;
  while (a <= -3.14159265) {
    a = a + TAU;
  }
  while (a > 3.14159265) {
    a = a - TAU;
  }
  return a;
}

fn shortest_angle_delta(currentAngle: f32, desiredAngle: f32) -> f32 {
  return normalize_angle(desiredAngle - currentAngle);
}

fn steer_towards(currentVel: vec2f, desiredDirection: vec2f, maxSpeed: f32, forceCap: f32) -> vec2f {
  let mag = length(desiredDirection);
  if (mag < 0.0001) {
    return vec2f(0.0, 0.0);
  }
  let desired = normalize(desiredDirection) * maxSpeed;
  let steer = desired - currentVel;
  return limit_magnitude(steer, forceCap);
}

fn hash_u32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash_u32(seed)) / 4294967295.0;
}

@compute @workgroup_size(${BOID_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let boidCount = u32(params.counts.x);
  let predatorCount = u32(params.counts.y);
  let index = gid.x;
  if (index >= boidCount) {
    return;
  }

  let worldW = params.world.x;
  let worldH = params.world.y;
  let dt = params.world.z;
  let frameScale = params.world.w;

  let perceptionSq = params.boidA.x * params.boidA.x;
  let separationSq = params.boidA.y * params.boidA.y;
  let predatorAvoidSq = params.boidA.z * params.boidA.z;
  let catchSq = params.boidA.w * params.boidA.w;

  let maxSpeed = params.boidB.x;
  let minSpeed = params.boidB.y;
  let maxForce = params.boidB.z;
  let maxTurnRate = params.boidB.w;

  let maxTurnAcceleration = params.boidC.x;
  let alignWeight = params.boidC.y;
  let cohesionWeight = params.boidC.z;
  let separationWeight = params.boidC.w;

  let predatorAvoidWeight = params.predatorA.x;

  var me = boidsIn[index];
  let mePos = me.posVel.xy;
  let meVel = me.posVel.zw;

  var align = vec2f(0.0, 0.0);
  var cohesion = vec2f(0.0, 0.0);
  var separation = vec2f(0.0, 0.0);
  var neighbors = 0u;

  for (var j: u32 = 0u; j < boidCount; j = j + 1u) {
    if (j == index) {
      continue;
    }

    let other = boidsIn[j];
    let dx = wrapped_delta(mePos.x, other.posVel.x, worldW);
    let dy = wrapped_delta(mePos.y, other.posVel.y, worldH);
    let delta = vec2f(dx, dy);
    let distSq = dot(delta, delta);

    if (distSq <= 0.0 || distSq > perceptionSq) {
      continue;
    }

    neighbors = neighbors + 1u;
    align = align + other.posVel.zw;
    cohesion = cohesion + delta;

    if (distSq < separationSq) {
      let invDist = inverseSqrt(max(distSq, 0.0001));
      separation = separation - delta * invDist;
    }
  }

  var predatorAvoid = vec2f(0.0, 0.0);
  var predatorThreats = 0u;

  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    let predator = predators[p];
    let dx = wrapped_delta(mePos.x, predator.posVel.x, worldW);
    let dy = wrapped_delta(mePos.y, predator.posVel.y, worldH);
    let delta = vec2f(dx, dy);
    let distSq = dot(delta, delta);

    if (distSq <= 0.0 || distSq > predatorAvoidSq) {
      continue;
    }

    let invDist = inverseSqrt(max(distSq, 0.0001));
    predatorAvoid = predatorAvoid - delta * invDist;
    predatorThreats = predatorThreats + 1u;
  }

  var desiredVel = meVel;

  if (neighbors > 0u) {
    let n = f32(neighbors);
    align = align / n;
    cohesion = cohesion / n;
    separation = separation / n;

    let alignSteer = steer_towards(meVel, align, maxSpeed, maxForce);
    let cohesionSteer = steer_towards(meVel, cohesion, maxSpeed, maxForce);
    let separationSteer = steer_towards(meVel, separation, maxSpeed, maxForce * 1.6);

    desiredVel = desiredVel + alignSteer * alignWeight;
    desiredVel = desiredVel + cohesionSteer * cohesionWeight;
    desiredVel = desiredVel + separationSteer * separationWeight;
  }

  if (predatorThreats > 0u) {
    let t = f32(predatorThreats);
    predatorAvoid = predatorAvoid / t;
    let predatorSteer = steer_towards(meVel, predatorAvoid, maxSpeed, maxForce * 2.2);
    desiredVel = desiredVel + predatorSteer * predatorAvoidWeight;
  }

  let desiredLimited = limit_magnitude(desiredVel, maxSpeed);
  let desiredSpeed = length(desiredLimited);
  let speed = clamp(desiredSpeed, minSpeed, maxSpeed);

  let currentHeading = me.headingTurn.x;
  let currentTurnRate = me.headingTurn.y;
  let targetHeading = select(currentHeading, atan2(desiredLimited.y, desiredLimited.x), desiredSpeed > 0.0001);
  let headingDelta = shortest_angle_delta(currentHeading, targetHeading);
  let desiredTurnRate = headingDelta / max(dt, 0.0001);
  let turnRateError = desiredTurnRate - currentTurnRate;
  let maxRateDelta = maxTurnAcceleration * dt;
  var nextTurnRate = currentTurnRate + clamp(turnRateError, -maxRateDelta, maxRateDelta);
  nextTurnRate = clamp(nextTurnRate, -maxTurnRate, maxTurnRate);
  let nextHeading = normalize_angle(currentHeading + nextTurnRate * dt);

  let nextVel = vec2f(cos(nextHeading), sin(nextHeading)) * speed;
  var nextPos = mePos + nextVel * frameScale;
  nextPos = vec2f(wrap_coordinate(nextPos.x, worldW), wrap_coordinate(nextPos.y, worldH));

  var caught = false;
  var caughtBy = 0u;
  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    let predator = predators[p];
    if (predator.headingTimers.z > 0.0) {
      continue;
    }
    let dx = wrapped_delta(nextPos.x, predator.posVel.x, worldW);
    let dy = wrapped_delta(nextPos.y, predator.posVel.y, worldH);
    if (dx * dx + dy * dy <= catchSq) {
      caught = true;
      caughtBy = p;
      break;
    }
  }

  var outState = BoidState(vec4f(nextPos, nextVel), vec4f(nextHeading, nextTurnRate, 0.0, 0.0));

  if (caught) {
    atomicStore(&caughtFlags[caughtBy], 1u);

    let frameSeed = u32(params.counts.z);
    let seed0 = frameSeed * 1664525u + (index + 1u) * 1013904223u + 17u;
    let seed1 = frameSeed * 22695477u + (index + 1u) * 747796405u + 31u;
    let seed2 = frameSeed * 1103515245u + (index + 1u) * 2891336453u + 71u;
    let seed3 = frameSeed * 2246822519u + (index + 1u) * 3266489917u + 127u;

    let rx = rand01(seed0);
    let ry = rand01(seed1);
    let rHeading = rand01(seed2) * TAU;
    let rSpeed = minSpeed + (maxSpeed - minSpeed) * rand01(seed3);

    let respawnPos = vec2f(rx * worldW, ry * worldH);
    let respawnVel = vec2f(cos(rHeading), sin(rHeading)) * rSpeed;
    outState = BoidState(vec4f(respawnPos, respawnVel), vec4f(rHeading, 0.0, 0.0, 0.0));
  }

  boidsOut[index] = outState;
}
`;

const predatorUpdateShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct PredatorState {
  posVel: vec4f,
  headingTimers: vec4f,
  aux: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

@group(0) @binding(0) var<storage, read> boidsIn: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> predators: array<PredatorState>;
@group(0) @binding(2) var<uniform> params: SimParams;

const MAX_PREDATORS: u32 = 16u;

fn wrap_coordinate(value: f32, size: f32) -> f32 {
  var result = value;
  if (result < 0.0) {
    result = result + size;
  }
  if (result >= size) {
    result = result - size;
  }
  return result;
}

fn wrapped_delta(srcValue: f32, dstValue: f32, size: f32) -> f32 {
  var delta = dstValue - srcValue;
  let half = size * 0.5;
  if (delta > half) {
    delta = delta - size;
  } else if (delta < -half) {
    delta = delta + size;
  }
  return delta;
}

const TAU: f32 = 6.28318530718;

fn normalize_angle(angle: f32) -> f32 {
  var a = angle;
  while (a <= -3.14159265) {
    a = a + TAU;
  }
  while (a > 3.14159265) {
    a = a - TAU;
  }
  return a;
}

fn shortest_angle_delta(currentAngle: f32, desiredAngle: f32) -> f32 {
  return normalize_angle(desiredAngle - currentAngle);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x > 0u) {
    return;
  }

  let worldW = params.world.x;
  let worldH = params.world.y;
  let dt = params.world.z;
  let frameScale = params.world.w;

  let boidCount = u32(params.counts.x);
  let predatorCount = u32(params.counts.y);

  let attentionSeconds = params.predatorA.y;
  let predatorSeparationRadiusSq = params.predatorA.z * params.predatorA.z;
  let predatorSeparationWeight = params.predatorA.w;

  let pauseSlowdownSeconds = max(params.predatorB.x, 0.0001);
  let predatorSpeed = params.boidB.x * params.predatorB.y;
  let predatorMaxTurnRate = params.predatorB.z;
  let predatorMaxTurnAcceleration = params.predatorB.w;

  var snapshot: array<PredatorState, MAX_PREDATORS>;

  for (var i: u32 = 0u; i < predatorCount; i = i + 1u) {
    snapshot[i] = predators[i];
  }

  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    var pred = snapshot[p];

    if (pred.headingTimers.z > 0.0) {
      let pauseStep = min(dt, pred.headingTimers.z);
      pred.headingTimers.z = max(0.0, pred.headingTimers.z - pauseStep);

      var pauseVel = vec2f(0.0, 0.0);
      if (pred.headingTimers.w > 0.0) {
        pred.headingTimers.w = max(0.0, pred.headingTimers.w - pauseStep);
        let slowdownFactor = pred.headingTimers.w / pauseSlowdownSeconds;
        pauseVel = pred.aux.xy * slowdownFactor;
        pred.posVel.x = wrap_coordinate(pred.posVel.x + pauseVel.x * (pauseStep * 60.0), worldW);
        pred.posVel.y = wrap_coordinate(pred.posVel.y + pauseVel.y * (pauseStep * 60.0), worldH);
      }

      pred.posVel.z = pauseVel.x;
      pred.posVel.w = pauseVel.y;
      pred.aux.w = 0.0;
      predators[p] = pred;
      continue;
    }

    pred.headingTimers.y = pred.headingTimers.y - dt;

    var targetIndex: i32 = i32(round(pred.aux.z));
    if (pred.headingTimers.y <= 0.0 || targetIndex < 0 || targetIndex >= i32(boidCount)) {
      var bestDistSq = 1e20;
      var closest: u32 = 0u;
      for (var b: u32 = 0u; b < boidCount; b = b + 1u) {
        let boid = boidsIn[b];
        let dx = wrapped_delta(pred.posVel.x, boid.posVel.x, worldW);
        let dy = wrapped_delta(pred.posVel.y, boid.posVel.y, worldH);
        let distSq = dx * dx + dy * dy;
        if (distSq < bestDistSq) {
          bestDistSq = distSq;
          closest = b;
        }
      }
      targetIndex = i32(closest);
      pred.headingTimers.y = attentionSeconds;
    }

    var chaseDir = vec2f(cos(pred.headingTimers.x), sin(pred.headingTimers.x));
    if (targetIndex >= 0 && targetIndex < i32(boidCount)) {
      let prey = boidsIn[u32(targetIndex)];
      let tx = wrapped_delta(pred.posVel.x, prey.posVel.x, worldW);
      let ty = wrapped_delta(pred.posVel.y, prey.posVel.y, worldH);
      let tLen = length(vec2f(tx, ty));
      if (tLen > 0.0001) {
        chaseDir = vec2f(tx, ty) / tLen;
      }
    }

    var separate = vec2f(0.0, 0.0);
    var separateCount = 0u;
    for (var q: u32 = 0u; q < predatorCount; q = q + 1u) {
      if (q == p) {
        continue;
      }
      let other = snapshot[q];
      let dx = wrapped_delta(pred.posVel.x, other.posVel.x, worldW);
      let dy = wrapped_delta(pred.posVel.y, other.posVel.y, worldH);
      let distSq = dx * dx + dy * dy;
      if (distSq <= 0.0 || distSq > predatorSeparationRadiusSq) {
        continue;
      }
      let invDist = inverseSqrt(max(distSq, 0.0001));
      separate = separate - vec2f(dx, dy) * invDist;
      separateCount = separateCount + 1u;
    }

    if (separateCount > 0u) {
      separate = separate / f32(separateCount);
    }

    var desiredDir = chaseDir + separate * predatorSeparationWeight;
    var desiredHeading = pred.headingTimers.x;
    if (length(desiredDir) > 0.0001) {
      desiredHeading = atan2(desiredDir.y, desiredDir.x);
    }

    let headingDelta = shortest_angle_delta(pred.headingTimers.x, desiredHeading);
    let desiredTurnRate = headingDelta / max(dt, 0.0001);
    let turnRateError = desiredTurnRate - pred.aux.w;
    let maxRateDelta = predatorMaxTurnAcceleration * dt;
    var nextTurnRate = pred.aux.w + clamp(turnRateError, -maxRateDelta, maxRateDelta);
    nextTurnRate = clamp(nextTurnRate, -predatorMaxTurnRate, predatorMaxTurnRate);
    let nextHeading = normalize_angle(pred.headingTimers.x + nextTurnRate * dt);

    pred.headingTimers.x = nextHeading;
    pred.aux.w = nextTurnRate;

    pred.posVel.z = cos(nextHeading) * predatorSpeed;
    pred.posVel.w = sin(nextHeading) * predatorSpeed;
    pred.posVel.x = wrap_coordinate(pred.posVel.x + pred.posVel.z * frameScale, worldW);
    pred.posVel.y = wrap_coordinate(pred.posVel.y + pred.posVel.w * frameScale, worldH);
    pred.aux.z = f32(targetIndex);

    predators[p] = pred;
  }
}
`;

const catchClearShader = `
struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

@group(0) @binding(0) var<storage, read_write> caughtFlags: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(${CATCH_CLEAR_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let predCount = u32(params.counts.y);
  if (gid.x >= predCount) {
    return;
  }
  atomicStore(&caughtFlags[gid.x], 0u);
}
`;

const predatorResolveShader = `
struct PredatorState {
  posVel: vec4f,
  headingTimers: vec4f,
  aux: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

@group(0) @binding(0) var<storage, read_write> predators: array<PredatorState>;
@group(0) @binding(1) var<storage, read_write> caughtFlags: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x > 0u) {
    return;
  }

  let predCount = u32(params.counts.y);

  for (var p: u32 = 0u; p < predCount; p = p + 1u) {
    let caught = atomicLoad(&caughtFlags[p]);
    if (caught == 0u) {
      continue;
    }

    var pred = predators[p];
    pred.headingTimers.z = 0.0;
    pred.headingTimers.w = 0.0;
    pred.aux.z = -1.0;
    pred.headingTimers.y = 0.0;

    predators[p] = pred;
  }
}
`;

const boidRenderShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<uniform> params: SimParams;

fn boid_vertex(vertexIndex: u32) -> vec2f {
  if (vertexIndex == 0u) {
    return vec2f(3.6, 0.0);
  }
  if (vertexIndex == 1u) {
    return vec2f(-2.1, 1.6);
  }
  return vec2f(-2.1, -1.6);
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VSOut {
  let boid = boids[instanceIndex];
  let heading = boid.headingTurn.x;
  let c = cos(heading);
  let s = sin(heading);

  let local = boid_vertex(vertexIndex);
  let rotated = vec2f(local.x * c - local.y * s, local.x * s + local.y * c);
  let world = boid.posVel.xy + rotated;

  let clipX = world.x / params.world.x * 2.0 - 1.0;
  let clipY = 1.0 - world.y / params.world.y * 2.0;

  let speedRatio = clamp(length(boid.posVel.zw) / max(params.boidB.x, 0.0001), 0.0, 1.0);
  let colorA = vec3f(0.40, 0.95, 1.00);
  let colorB = vec3f(0.66, 0.98, 0.90);

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.color = mix(colorA, colorB, speedRatio);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

const boidHeatmapRenderShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) radial: vec2f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<uniform> params: SimParams;

fn quad_vertex(vertexIndex: u32) -> vec2f {
  if (vertexIndex == 0u) {
    return vec2f(-1.0, -1.0);
  }
  if (vertexIndex == 1u) {
    return vec2f(1.0, -1.0);
  }
  if (vertexIndex == 2u) {
    return vec2f(-1.0, 1.0);
  }
  if (vertexIndex == 3u) {
    return vec2f(-1.0, 1.0);
  }
  if (vertexIndex == 4u) {
    return vec2f(1.0, -1.0);
  }
  return vec2f(1.0, 1.0);
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VSOut {
  let boid = boids[instanceIndex];
  let radial = quad_vertex(vertexIndex);
  let kernelRadius = 11.0;
  let world = boid.posVel.xy + radial * kernelRadius;

  let clipX = world.x / params.world.x * 2.0 - 1.0;
  let clipY = 1.0 - world.y / params.world.y * 2.0;

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.radial = radial;
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let distSq = dot(in.radial, in.radial);
  if (distSq > 1.0) {
    discard;
  }

  let core = exp(-distSq * 2.35);
  let halo = exp(-distSq * 0.95);
  let localDensity = core * 0.72 + halo * 0.28;
  let contribution = localDensity * 0.082;

  let dark = vec3f(0.005, 0.015, 0.055);
  let low = vec3f(0.06, 0.25, 0.75);
  let mid = vec3f(0.24, 0.72, 1.00);
  let hot = vec3f(1.00, 0.86, 0.20);
  let peak = vec3f(1.00, 1.00, 0.94);

  let rampA = mix(dark, low, smoothstep(0.04, 0.32, localDensity));
  let rampB = mix(low, mid, smoothstep(0.20, 0.62, localDensity));
  let rampC = mix(mid, hot, smoothstep(0.52, 0.90, localDensity));
  var color = mix(rampA, rampB, smoothstep(0.18, 0.58, localDensity));
  color = mix(color, rampC, smoothstep(0.48, 0.88, localDensity));
  color = mix(color, peak, smoothstep(0.90, 1.0, localDensity));
  return vec4f(color * contribution, contribution);
}
`;

const predatorRenderShader = `
struct PredatorState {
  posVel: vec4f,
  headingTimers: vec4f,
  aux: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(1) var<uniform> params: SimParams;

fn predator_vertex(vertexIndex: u32) -> vec2f {
  if (vertexIndex == 0u) {
    return vec2f(6.2, 0.0);
  }
  if (vertexIndex == 1u) {
    return vec2f(-3.6, 2.4);
  }
  return vec2f(-3.6, -2.4);
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VSOut {
  let predator = predators[instanceIndex];
  let heading = predator.headingTimers.x;
  let c = cos(heading);
  let s = sin(heading);

  let local = predator_vertex(vertexIndex);
  let rotated = vec2f(local.x * c - local.y * s, local.x * s + local.y * c);
  let world = predator.posVel.xy + rotated;

  let clipX = world.x / params.world.x * 2.0 - 1.0;
  let clipY = 1.0 - world.y / params.world.y * 2.0;

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  return out;
}

@fragment
fn fsMain() -> @location(0) vec4f {
  return vec4f(1.0, 0.32, 0.38, 1.0);
}
`;

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function createInitialBoidData() {
  const data = new Float32Array(BOID_COUNT * BOID_FLOATS);
  for (let i = 0; i < BOID_COUNT; i += 1) {
    const heading = rand(0, Math.PI * 2);
    const speed = rand(config.minSpeed, config.maxSpeed);
    const base = i * BOID_FLOATS;

    data[base + 0] = rand(0, worldWidth);
    data[base + 1] = rand(0, worldHeight);
    data[base + 2] = Math.cos(heading) * speed;
    data[base + 3] = Math.sin(heading) * speed;
    data[base + 4] = heading;
    data[base + 5] = 0;
    data[base + 6] = 0;
    data[base + 7] = 0;
  }
  return data;
}

function createInitialPredatorData() {
  const data = new Float32Array(PREDATOR_COUNT * PREDATOR_FLOATS);
  const predatorSpeed = config.maxSpeed * config.predatorSpeedFactor;

  for (let i = 0; i < PREDATOR_COUNT; i += 1) {
    const heading = rand(0, Math.PI * 2);
    const base = i * PREDATOR_FLOATS;

    data[base + 0] = rand(0, worldWidth);
    data[base + 1] = rand(0, worldHeight);
    data[base + 2] = Math.cos(heading) * predatorSpeed;
    data[base + 3] = Math.sin(heading) * predatorSpeed;

    data[base + 4] = heading;
    data[base + 5] = 0;
    data[base + 6] = 0;
    data[base + 7] = 0;

    data[base + 8] = 0;
    data[base + 9] = 0;
    data[base + 10] = -1;
    data[base + 11] = 0;
  }

  return data;
}

function updateTextState(mode = 'running') {
  lastTextState = JSON.stringify({
    mode,
    renderer: 'webgpu',
    coordinateSystem: 'origin top-left, +x right, +y down',
    viewport: { width: worldWidth, height: worldHeight },
    boidCount: BOID_COUNT,
    predatorCount: PREDATOR_COUNT,
    frameIndex,
    viewMode: renderMode,
    maxTurnAccelerationDegPerSec2: Number(turnAccelRange.value),
    minSpeed: Number(minSpeedRange.value),
    predatorAttentionSeconds: Number(predatorAttentionRange.value),
    predatorSpeedFactor: config.predatorSpeedFactor,
  });
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  worldWidth = Math.max(1, Math.floor(rect.width));
  worldHeight = Math.max(1, Math.floor(rect.height));

  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  devicePixelRatioCached = dpr;
  canvas.width = Math.floor(worldWidth * dpr);
  canvas.height = Math.floor(worldHeight * dpr);
}

function createParamsArray(dtSeconds) {
  const frameScale = dtSeconds * 60;
  const predatorMaxTurnRate = config.maxTurnRate * config.predatorTurnRateFactor;
  const predatorMaxTurnAcceleration = config.maxTurnAcceleration * config.predatorTurnAccelerationFactor;
  return new Float32Array([
    worldWidth, worldHeight, dtSeconds, frameScale,
    BOID_COUNT, PREDATOR_COUNT, frameIndex, 0,
    config.perceptionRadius, config.separationRadius, config.predatorAvoidRadius, config.predatorCatchRadius,
    config.maxSpeed, config.minSpeed, config.maxForce, config.maxTurnRate,
    config.maxTurnAcceleration, config.alignWeight, config.cohesionWeight, config.separationWeight,
    config.predatorAvoidWeight, config.predatorAttentionSeconds, config.predatorSeparationRadius, config.predatorSeparationWeight,
    config.predatorPauseSlowdownSeconds, config.predatorSpeedFactor,
    predatorMaxTurnRate, predatorMaxTurnAcceleration,
  ]);
}

function writeParams(dtSeconds) {
  const params = createParamsArray(dtSeconds);
  gpu.device.queue.writeBuffer(gpu.paramsBuffer, 0, params);
}

function reseedSimulation() {
  if (!gpu) {
    return;
  }

  const boidData = createInitialBoidData();
  const predatorData = createInitialPredatorData();
  const zeroFlags = new Uint32Array(PREDATOR_COUNT);

  gpu.device.queue.writeBuffer(gpu.boidBuffers[0], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.boidBuffers[1], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.predatorBuffer, 0, predatorData);
  gpu.device.queue.writeBuffer(gpu.caughtFlagsBuffer, 0, zeroFlags);

  frameIndex = 0;
  accumulator = 0;
  currentBoidBufferIndex = 0;
  writeParams(FIXED_STEP);
  updateTextState();
}

function stepSimulation(dtSeconds) {
  if (!gpu) {
    return;
  }

  const boidUpdateBindGroup = gpu.boidUpdateBindGroups[currentBoidBufferIndex];
  const predatorUpdateBindGroup = gpu.predatorUpdateBindGroups[currentBoidBufferIndex];
  if (!boidUpdateBindGroup || !predatorUpdateBindGroup) {
    throw new Error(`Missing compute bind group for boid buffer index ${currentBoidBufferIndex}.`);
  }

  writeParams(dtSeconds);

  const encoder = gpu.device.createCommandEncoder();
  const computePass = encoder.beginComputePass();

  try {
    computePass.setPipeline(gpu.clearCaughtPipeline);
    computePass.setBindGroup(0, gpu.clearCaughtBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(PREDATOR_COUNT / CATCH_CLEAR_WORKGROUP_SIZE));

    computePass.setPipeline(gpu.predatorUpdatePipeline);
    computePass.setBindGroup(0, predatorUpdateBindGroup);
    computePass.dispatchWorkgroups(1);

    computePass.setPipeline(gpu.boidUpdatePipeline);
    computePass.setBindGroup(0, boidUpdateBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(BOID_COUNT / BOID_WORKGROUP_SIZE));

    computePass.setPipeline(gpu.predatorResolvePipeline);
    computePass.setBindGroup(0, gpu.predatorResolveBindGroup);
    computePass.dispatchWorkgroups(1);
  } finally {
    computePass.end();
  }

  gpu.device.queue.submit([encoder.finish()]);

  currentBoidBufferIndex = 1 - currentBoidBufferIndex;
  frameIndex += 1;
}

function renderFrame() {
  if (!gpu) {
    return;
  }

  const boidRenderBindGroup = gpu.boidRenderBindGroups[currentBoidBufferIndex];
  const heatmapRenderBindGroup = gpu.boidHeatmapRenderBindGroups[currentBoidBufferIndex];
  if (!boidRenderBindGroup || !heatmapRenderBindGroup) {
    throw new Error(`Missing render bind group for boid buffer index ${currentBoidBufferIndex}.`);
  }

  const encoder = gpu.device.createCommandEncoder();
  const view = gpu.context.getCurrentTexture().createView();

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.03, g: 0.07, b: 0.16, a: 1 },
      },
    ],
  });

  try {
    if (renderMode === 'heatmap') {
      renderPass.setPipeline(gpu.boidHeatmapRenderPipeline);
      renderPass.setBindGroup(0, heatmapRenderBindGroup);
      renderPass.draw(6, BOID_COUNT, 0, 0);
    } else {
      renderPass.setPipeline(gpu.boidRenderPipeline);
      renderPass.setBindGroup(0, boidRenderBindGroup);
      renderPass.draw(3, BOID_COUNT, 0, 0);
      renderPass.setPipeline(gpu.predatorRenderPipeline);
      renderPass.setBindGroup(0, gpu.predatorRenderBindGroup);
      renderPass.draw(3, PREDATOR_COUNT, 0, 0);
    }
  } finally {
    renderPass.end();
  }

  gpu.device.queue.submit([encoder.finish()]);
}

function tick(timestamp) {
  if (!gpu || gpuRuntimeError) {
    return;
  }

  try {
    if (lastFrameTime === 0) {
      lastFrameTime = timestamp;
    }

    const rawDelta = (timestamp - lastFrameTime) / 1000;
    lastFrameTime = timestamp;
    accumulator += Math.min(rawDelta, 0.05);

    while (accumulator >= FIXED_STEP) {
      stepSimulation(FIXED_STEP);
      accumulator -= FIXED_STEP;
    }

    renderFrame();
    updateTextState();
    requestAnimationFrame(tick);
  } catch (error) {
    gpuRuntimeError = true;
    console.error(error);
    setHeaderStatus(`WebGPU runtime error: ${error.message}`);
    updateTextState('webgpu_runtime_error');
  }
}

function advanceTime(ms) {
  if (!gpu) {
    return;
  }

  const total = Math.max(0, ms / 1000);
  const fullSteps = Math.floor(total / FIXED_STEP);
  const remainder = total - fullSteps * FIXED_STEP;

  for (let i = 0; i < fullSteps; i += 1) {
    stepSimulation(FIXED_STEP);
  }
  if (remainder > 0) {
    stepSimulation(remainder);
  }

  renderFrame();
  updateTextState();
}

function setHeaderStatus(text) {
  const subtitle = document.querySelector('header p');
  if (subtitle) {
    subtitle.textContent = text;
  }
}

function updateViewToggleLabel() {
  if (!viewToggleButton) {
    return;
  }
  const isHeatmap = renderMode === 'heatmap';
  viewToggleButton.textContent = isHeatmap ? 'View: Heatmap' : 'View: Arrows';
  viewToggleButton.setAttribute('aria-pressed', isHeatmap ? 'true' : 'false');
}

async function initWebGPU() {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No compatible GPU adapter found.');
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Failed to acquire WebGPU canvas context.');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const boidBufferBytes = BOID_COUNT * BOID_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const predatorBufferBytes = PREDATOR_COUNT * PREDATOR_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const paramsBufferBytes = 28 * Float32Array.BYTES_PER_ELEMENT;
  const caughtFlagsBytes = PREDATOR_COUNT * Uint32Array.BYTES_PER_ELEMENT;

  const boidBuffers = [
    device.createBuffer({
      size: boidBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      size: boidBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];

  const predatorBuffer = device.createBuffer({
    size: predatorBufferBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const caughtFlagsBuffer = device.createBuffer({
    size: caughtFlagsBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const paramsBuffer = device.createBuffer({
    size: paramsBufferBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  async function createCheckedShaderModule(label, code) {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((msg) => msg.type === 'error');
    if (errors.length > 0) {
      const details = errors
        .map((msg) => `[${msg.lineNum}:${msg.linePos}] ${msg.message}`)
        .join('\n');
      throw new Error(`${label} WGSL compilation failed:\n${details}`);
    }
    return module;
  }

  const boidUpdateModule = await createCheckedShaderModule('boidUpdateShader', boidUpdateShader);
  const predatorUpdateModule = await createCheckedShaderModule('predatorUpdateShader', predatorUpdateShader);
  const clearCaughtModule = await createCheckedShaderModule('catchClearShader', catchClearShader);
  const predatorResolveModule = await createCheckedShaderModule('predatorResolveShader', predatorResolveShader);
  const boidRenderModule = await createCheckedShaderModule('boidRenderShader', boidRenderShader);
  const boidHeatmapRenderModule = await createCheckedShaderModule('boidHeatmapRenderShader', boidHeatmapRenderShader);
  const predatorRenderModule = await createCheckedShaderModule('predatorRenderShader', predatorRenderShader);

  const boidUpdatePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: boidUpdateModule,
      entryPoint: 'main',
    },
  });

  const predatorUpdatePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: predatorUpdateModule,
      entryPoint: 'main',
    },
  });

  const clearCaughtPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: clearCaughtModule,
      entryPoint: 'main',
    },
  });

  const predatorResolvePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: predatorResolveModule,
      entryPoint: 'main',
    },
  });

  const boidRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: boidRenderModule,
      entryPoint: 'vsMain',
    },
    fragment: {
      module: boidRenderModule,
      entryPoint: 'fsMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
  });

  const boidHeatmapRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: boidHeatmapRenderModule,
      entryPoint: 'vsMain',
    },
    fragment: {
      module: boidHeatmapRenderModule,
      entryPoint: 'fsMain',
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
  });

  const predatorRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: predatorRenderModule,
      entryPoint: 'vsMain',
    },
    fragment: {
      module: predatorRenderModule,
      entryPoint: 'fsMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
  });

  const boidUpdateBindGroups = [
    device.createBindGroup({
      layout: boidUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[0] } },
        { binding: 1, resource: { buffer: boidBuffers[1] } },
        { binding: 2, resource: { buffer: predatorBuffer } },
        { binding: 3, resource: { buffer: caughtFlagsBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: boidUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[1] } },
        { binding: 1, resource: { buffer: boidBuffers[0] } },
        { binding: 2, resource: { buffer: predatorBuffer } },
        { binding: 3, resource: { buffer: caughtFlagsBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const predatorUpdateBindGroups = [
    device.createBindGroup({
      layout: predatorUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[0] } },
        { binding: 1, resource: { buffer: predatorBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: predatorUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[1] } },
        { binding: 1, resource: { buffer: predatorBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const clearCaughtBindGroup = device.createBindGroup({
    layout: clearCaughtPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: caughtFlagsBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  const predatorResolveBindGroup = device.createBindGroup({
    layout: predatorResolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: predatorBuffer } },
      { binding: 1, resource: { buffer: caughtFlagsBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  const boidRenderBindGroups = [
    device.createBindGroup({
      layout: boidRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[0] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: boidRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[1] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const boidHeatmapRenderBindGroups = [
    device.createBindGroup({
      layout: boidHeatmapRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[0] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: boidHeatmapRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[1] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const predatorRenderBindGroup = device.createBindGroup({
    layout: predatorRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: predatorBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  gpu = {
    device,
    context,
    format,
    boidBuffers,
    predatorBuffer,
    caughtFlagsBuffer,
    paramsBuffer,
    boidUpdatePipeline,
    predatorUpdatePipeline,
    clearCaughtPipeline,
    predatorResolvePipeline,
    boidRenderPipeline,
    boidHeatmapRenderPipeline,
    predatorRenderPipeline,
    boidUpdateBindGroups,
    predatorUpdateBindGroups,
    clearCaughtBindGroup,
    predatorResolveBindGroup,
    boidRenderBindGroups,
    boidHeatmapRenderBindGroups,
    predatorRenderBindGroup,
  };

  reseedSimulation();
  updateTextState();
}

function handleResize() {
  resizeCanvas();
  if (gpu) {
    gpu.context.configure({
      device: gpu.device,
      format: gpu.format,
      alphaMode: 'opaque',
    });
    writeParams(FIXED_STEP);
  }
}

turnAccelRange.addEventListener('input', () => {
  turnAccelValue.textContent = turnAccelRange.value;
  config.maxTurnAcceleration = (Number(turnAccelRange.value) * Math.PI) / 180;
});

minSpeedRange.addEventListener('input', () => {
  minSpeedValue.textContent = Number(minSpeedRange.value).toFixed(2);
  config.minSpeed = Math.min(Number(minSpeedRange.value), config.maxSpeed);
});

predatorAttentionRange.addEventListener('input', () => {
  predatorAttentionValue.textContent = Number(predatorAttentionRange.value).toFixed(1);
  config.predatorAttentionSeconds = Number(predatorAttentionRange.value);
});

viewToggleButton.addEventListener('click', () => {
  renderMode = renderMode === 'boids' ? 'heatmap' : 'boids';
  updateViewToggleLabel();
  updateTextState();
});

restartButton.addEventListener('click', () => {
  reseedSimulation();
});

window.addEventListener('resize', handleResize);
window.render_game_to_text = () => lastTextState;
window.advanceTime = advanceTime;

turnAccelValue.textContent = turnAccelRange.value;
minSpeedValue.textContent = Number(minSpeedRange.value).toFixed(2);
predatorAttentionValue.textContent = Number(predatorAttentionRange.value).toFixed(1);
updateViewToggleLabel();

resizeCanvas();
updateTextState('initializing');

initWebGPU()
  .then(() => {
    setHeaderStatus(`${BOID_COUNT} boids, WebGPU compute + rendering active.`);
    gpu.device.addEventListener('uncapturederror', (event) => {
      gpuRuntimeError = true;
      console.error(event.error);
      setHeaderStatus(`WebGPU uncaptured error: ${event.error.message}`);
      updateTextState('webgpu_uncaptured_error');
    });
    gpu.device.lost.then((info) => {
      gpuRuntimeError = true;
      setHeaderStatus(`WebGPU device lost: ${info.message}`);
      updateTextState('webgpu_device_lost');
    });
    requestAnimationFrame(tick);
  })
  .catch((error) => {
    console.error(error);
    setHeaderStatus(`WebGPU init failed: ${error.message}`);
    updateTextState('webgpu_init_failed');
  });
