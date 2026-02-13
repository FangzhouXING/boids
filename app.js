const canvas = document.getElementById('boidCanvas');
const restartButton = document.getElementById('restartButton');
const turnAccelRange = document.getElementById('turnAccelRange');
const turnAccelValue = document.getElementById('turnAccelValue');
const minSpeedRange = document.getElementById('minSpeedRange');
const minSpeedValue = document.getElementById('minSpeedValue');
const predatorAttentionRange = document.getElementById('predatorAttentionRange');
const predatorAttentionValue = document.getElementById('predatorAttentionValue');
const viewToggleButton = document.getElementById('viewToggleButton');
const heatmapSpacingRange = document.getElementById('heatmapSpacingRange');
const heatmapSpacingValue = document.getElementById('heatmapSpacingValue');
const heatmapDotRadiusRange = document.getElementById('heatmapDotRadiusRange');
const heatmapDotRadiusValue = document.getElementById('heatmapDotRadiusValue');
const heatmapDiffuseRange = document.getElementById('heatmapDiffuseRange');
const heatmapDiffuseValue = document.getElementById('heatmapDiffuseValue');
const heatmapSamplesRange = document.getElementById('heatmapSamplesRange');
const heatmapSamplesValue = document.getElementById('heatmapSamplesValue');
const heatmapTrendGainRange = document.getElementById('heatmapTrendGainRange');
const heatmapTrendGainValue = document.getElementById('heatmapTrendGainValue');
const heatmapTrendDeadbandRange = document.getElementById('heatmapTrendDeadbandRange');
const heatmapTrendDeadbandValue = document.getElementById('heatmapTrendDeadbandValue');
const pherTrailWeightRange = document.getElementById('pherTrailWeightRange');
const pherTrailWeightValue = document.getElementById('pherTrailWeightValue');
const pherFearWeightRange = document.getElementById('pherFearWeightRange');
const pherFearWeightValue = document.getElementById('pherFearWeightValue');
const pherDiffusionRange = document.getElementById('pherDiffusionRange');
const pherDiffusionValue = document.getElementById('pherDiffusionValue');
const pherDecayRange = document.getElementById('pherDecayRange');
const pherDecayValue = document.getElementById('pherDecayValue');
const panicBoostRange = document.getElementById('panicBoostRange');
const panicBoostValue = document.getElementById('panicBoostValue');
const fpsValue = document.getElementById('fpsValue');
const frameCpuValue = document.getElementById('frameCpuValue');
const simCpuValue = document.getElementById('simCpuValue');
const heatmapCpuValue = document.getElementById('heatmapCpuValue');
const renderCpuValue = document.getElementById('renderCpuValue');

const FIXED_STEP = 1 / 60;
const BOID_COUNT = 30000;
const PREDATOR_COUNT = 5;
const BOID_WORKGROUP_SIZE = 128;
const CATCH_CLEAR_WORKGROUP_SIZE = 32;
const HEATMAP_WORKGROUP_SIZE = 128;
const GRID_WORKGROUP_SIZE = 128;
const BOID_FLOATS = 8;
const PREDATOR_FLOATS = 12;
const PHEROMONE_FLOATS = 4;
const GRID_CELL_SIZE = 72;
const GRID_MAX_CELLS = 2048;
const GRID_CELL_CAPACITY = 256;
const HEATMAP_DOT_SPACING = 8;
const HEATMAP_DOT_RADIUS = 3.5;
const HEATMAP_DIFFUSE_RADIUS = 22;
const HEATMAP_SAMPLE_BUDGET = 320;
const HEATMAP_TREND_GAIN = 4.2;
const HEATMAP_TREND_DEADBAND = 0.045;
const HEATMAP_MAX_POINTS = 65536;

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
  heatmapDotSpacingPx: Number(heatmapSpacingRange?.value || HEATMAP_DOT_SPACING),
  heatmapDotRadiusPx: Number(heatmapDotRadiusRange?.value || HEATMAP_DOT_RADIUS),
  heatmapDiffuseRadiusPx: Number(heatmapDiffuseRange?.value || HEATMAP_DIFFUSE_RADIUS),
  heatmapSampleBudget: Number(heatmapSamplesRange?.value || HEATMAP_SAMPLE_BUDGET),
  heatmapTrendGain: Number(heatmapTrendGainRange?.value || HEATMAP_TREND_GAIN),
  heatmapTrendDeadband: Number(heatmapTrendDeadbandRange?.value || HEATMAP_TREND_DEADBAND),
  pherTrailWeight: Number(pherTrailWeightRange?.value || 0.6),
  pherFearWeight: Number(pherFearWeightRange?.value || 1.5),
  pherDiffusion: Number(pherDiffusionRange?.value || 0.28),
  pherDecay: Number(pherDecayRange?.value || 0.3),
  panicBoost: Number(panicBoostRange?.value || 1.2),
};

let worldWidth = 960;
let worldHeight = 560;
let devicePixelRatioCached = Math.max(window.devicePixelRatio || 1, 1);
let frameIndex = 0;
let accumulator = 0;
let lastFrameTime = 0;
let currentBoidBufferIndex = 0;
let renderMode = 'boids';
let heatmapPointCount = 1;
let currentHeatmapBufferIndex = 0;
let currentPheromoneBufferIndex = 0;
let gpu = null;
let gpuRuntimeError = false;
let lastTextState = JSON.stringify({
  mode: 'initializing',
  boidCount: BOID_COUNT,
  predatorCount: PREDATOR_COUNT,
});
const perfStats = {
  fps: 0,
  frameCpuMs: 0,
  simCpuMs: 0,
  heatmapCpuMs: 0,
  renderCpuMs: 0,
  alpha: 0.15,
  lastPanelUpdate: 0,
};

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

struct PheromoneSample {
  values: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
};

@group(0) @binding(0) var<storage, read> boidsIn: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> boidsOut: array<BoidState>;
@group(0) @binding(2) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(3) var<storage, read> pheromones: array<PheromoneSample>;
@group(0) @binding(4) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> cellBoids: array<u32>;
@group(0) @binding(6) var<storage, read_write> caughtFlags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: SimParams;

const TAU: f32 = 6.28318530718;
const GRID_CELL_SIZE: f32 = ${GRID_CELL_SIZE.toFixed(1)};
const GRID_MAX_CELLS: u32 = ${GRID_MAX_CELLS}u;
const GRID_CELL_CAPACITY: u32 = ${GRID_CELL_CAPACITY}u;
const PHEROMONE_MAX_POINTS: u32 = ${HEATMAP_MAX_POINTS}u;

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

fn wrap_index(value: i32, size: i32) -> u32 {
  var v = value % size;
  if (v < 0) {
    v = v + size;
  }
  return u32(v);
}

fn read_pheromone_sample(x: i32, y: i32, cols: u32, rows: u32, sampleCount: u32) -> vec2f {
  if (cols == 0u || rows == 0u || sampleCount == 0u) {
    return vec2f(0.0, 0.0);
  }
  let sx = wrap_index(x, i32(cols));
  let sy = wrap_index(y, i32(rows));
  let idx = sy * cols + sx;
  if (idx >= sampleCount || idx >= PHEROMONE_MAX_POINTS) {
    return vec2f(0.0, 0.0);
  }
  return pheromones[idx].values.xy;
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
  let cols = max(1u, u32(ceil(worldW / GRID_CELL_SIZE)));
  let rows = max(1u, u32(ceil(worldH / GRID_CELL_SIZE)));
  let gridCellCount = min(GRID_MAX_CELLS, cols * rows);
  let pherSpacing = max(params.heatmapA.x, 0.75);
  let pherCols = max(1u, u32(floor(worldW / pherSpacing)));
  let pherRows = max(1u, u32(floor(worldH / pherSpacing)));
  let pherCount = min(PHEROMONE_MAX_POINTS, pherCols * pherRows);

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
  let trailWeight = params.lifeA.x;
  let fearWeight = params.lifeA.y;
  let panicBoost = params.lifeA.z;

  var me = boidsIn[index];
  let mePos = me.posVel.xy;
  let meVel = me.posVel.zw;
  let meCellX = i32(min(cols - 1u, u32(floor(mePos.x / GRID_CELL_SIZE))));
  let meCellY = i32(min(rows - 1u, u32(floor(mePos.y / GRID_CELL_SIZE))));
  let cellRange = i32(max(1.0, ceil(params.boidA.x / GRID_CELL_SIZE)));

  var align = vec2f(0.0, 0.0);
  var cohesion = vec2f(0.0, 0.0);
  var separation = vec2f(0.0, 0.0);
  var neighbors = 0u;

  let pherX = i32(min(pherCols - 1u, u32(floor(mePos.x / pherSpacing))));
  let pherY = i32(min(pherRows - 1u, u32(floor(mePos.y / pherSpacing))));
  let pherCenter = read_pheromone_sample(pherX, pherY, pherCols, pherRows, pherCount);
  let pherRight = read_pheromone_sample(pherX + 1, pherY, pherCols, pherRows, pherCount);
  let pherLeft = read_pheromone_sample(pherX - 1, pherY, pherCols, pherRows, pherCount);
  let pherDown = read_pheromone_sample(pherX, pherY + 1, pherCols, pherRows, pherCount);
  let pherUp = read_pheromone_sample(pherX, pherY - 1, pherCols, pherRows, pherCount);
  let trailGradient = vec2f(pherRight.x - pherLeft.x, pherDown.x - pherUp.x);
  let fearGradient = vec2f(pherRight.y - pherLeft.y, pherDown.y - pherUp.y);
  var panic = clamp(pherCenter.y * panicBoost, 0.0, 1.0);

  for (var oy: i32 = -cellRange; oy <= cellRange; oy = oy + 1) {
    let ny = wrap_index(meCellY + oy, i32(rows));
    for (var ox: i32 = -cellRange; ox <= cellRange; ox = ox + 1) {
      let nx = wrap_index(meCellX + ox, i32(cols));
      let cellIndex = ny * cols + nx;
      if (cellIndex >= gridCellCount) {
        continue;
      }

      let count = min(atomicLoad(&cellCounts[cellIndex]), GRID_CELL_CAPACITY);
      for (var n: u32 = 0u; n < count; n = n + 1u) {
        let j = cellBoids[cellIndex * GRID_CELL_CAPACITY + n];
        if (j == index || j >= boidCount) {
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

    let alignScale = mix(1.0, 0.55, panic);
    let cohesionScale = mix(1.0, 0.45, panic);
    let separationScale = 1.0 + panic * 1.65;

    let alignSteer = steer_towards(meVel, align, maxSpeed, maxForce);
    let cohesionSteer = steer_towards(meVel, cohesion, maxSpeed, maxForce);
    let separationSteer = steer_towards(meVel, separation, maxSpeed, maxForce * 1.6);

    desiredVel = desiredVel + alignSteer * alignWeight * alignScale;
    desiredVel = desiredVel + cohesionSteer * cohesionWeight * cohesionScale;
    desiredVel = desiredVel + separationSteer * separationWeight * separationScale;
  }

  if (predatorThreats > 0u) {
    let t = f32(predatorThreats);
    predatorAvoid = predatorAvoid / t;
    let predatorSteer = steer_towards(meVel, predatorAvoid, maxSpeed, maxForce * 2.2);
    desiredVel = desiredVel + predatorSteer * predatorAvoidWeight * (1.0 + panic * 1.6);
    panic = max(panic, clamp(f32(predatorThreats) * 0.45, 0.0, 1.0));
  }

  if (dot(trailGradient, trailGradient) > 0.000001) {
    let trailSteer = steer_towards(meVel, trailGradient, maxSpeed, maxForce);
    desiredVel = desiredVel + trailSteer * trailWeight * (1.0 - panic);
  }

  if (dot(fearGradient, fearGradient) > 0.000001) {
    let fearSteer = steer_towards(meVel, -fearGradient, maxSpeed, maxForce * 1.7);
    desiredVel = desiredVel + fearSteer * fearWeight;
  }

  let desiredLimited = limit_magnitude(desiredVel, maxSpeed);
  let desiredSpeed = length(desiredLimited);
  let panicMinSpeed = min(maxSpeed, minSpeed + panic * 0.7);
  let speed = clamp(desiredSpeed, panicMinSpeed, maxSpeed);

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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
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

const gridClearShader = `
@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;

@compute @workgroup_size(${GRID_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${GRID_MAX_CELLS}u) {
    return;
  }
  atomicStore(&cellCounts[gid.x], 0u);
}
`;

const gridBuildShader = `
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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cellBoids: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

const GRID_CELL_SIZE: f32 = ${GRID_CELL_SIZE.toFixed(1)};
const GRID_MAX_CELLS: u32 = ${GRID_MAX_CELLS}u;
const GRID_CELL_CAPACITY: u32 = ${GRID_CELL_CAPACITY}u;

@compute @workgroup_size(${BOID_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let boidCount = u32(params.counts.x);
  let index = gid.x;
  if (index >= boidCount) {
    return;
  }

  let worldW = max(params.world.x, 1.0);
  let worldH = max(params.world.y, 1.0);
  let cols = max(1u, u32(ceil(worldW / GRID_CELL_SIZE)));
  let rows = max(1u, u32(ceil(worldH / GRID_CELL_SIZE)));
  let cellCount = min(GRID_MAX_CELLS, cols * rows);
  if (cellCount == 0u) {
    return;
  }

  let boid = boids[index];
  let cx = min(cols - 1u, u32(floor(boid.posVel.x / GRID_CELL_SIZE)));
  let cy = min(rows - 1u, u32(floor(boid.posVel.y / GRID_CELL_SIZE)));
  let rawCellIndex = cy * cols + cx;
  let cellIndex = min(rawCellIndex, cellCount - 1u);
  let slot = atomicAdd(&cellCounts[cellIndex], 1u);
  if (slot < GRID_CELL_CAPACITY) {
    cellBoids[cellIndex * GRID_CELL_CAPACITY + slot] = index;
  }
}
`;

const pheromoneUpdateShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct PredatorState {
  posVel: vec4f,
  headingTimers: vec4f,
  aux: vec4f,
};

struct PheromoneSample {
  values: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(2) var<storage, read> pheromonePrev: array<PheromoneSample>;
@group(0) @binding(3) var<storage, read_write> pheromoneNext: array<PheromoneSample>;
@group(0) @binding(4) var<uniform> params: SimParams;

const PHEROMONE_MAX_POINTS: u32 = ${HEATMAP_MAX_POINTS}u;
const MAX_BOID_SAMPLES: u32 = 192u;

fn wrap_index(value: i32, size: i32) -> u32 {
  var v = value % size;
  if (v < 0) {
    v = v + size;
  }
  return u32(v);
}

fn wrapped_abs_delta(a: f32, b: f32, size: f32) -> f32 {
  let direct = abs(a - b);
  return min(direct, size - direct);
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

fn read_pheromone(x: i32, y: i32, cols: u32, rows: u32, sampleCount: u32) -> vec2f {
  if (sampleCount == 0u || cols == 0u || rows == 0u) {
    return vec2f(0.0, 0.0);
  }
  let sx = wrap_index(x, i32(cols));
  let sy = wrap_index(y, i32(rows));
  let idx = sy * cols + sx;
  if (idx >= sampleCount || idx >= PHEROMONE_MAX_POINTS) {
    return vec2f(0.0, 0.0);
  }
  return pheromonePrev[idx].values.xy;
}

@compute @workgroup_size(${HEATMAP_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let sampleCount = u32(params.counts.w);
  if (idx >= sampleCount || idx >= PHEROMONE_MAX_POINTS) {
    return;
  }

  let worldW = params.world.x;
  let worldH = params.world.y;
  let boidCount = max(1u, u32(params.counts.x));
  let predatorCount = u32(params.counts.y);
  let dt = params.world.z;
  let spacing = max(params.heatmapA.x, 0.75);
  let diffuseRadius = max(params.heatmapA.z, 2.0);
  let diffuseRadiusSq = diffuseRadius * diffuseRadius;
  let cols = max(1u, u32(floor(worldW / spacing)));
  let rows = max(1u, u32(floor(worldH / spacing)));
  let sx = idx % cols;
  let sy = idx / cols;
  let samplePos = vec2f((f32(sx) + 0.5) * spacing, (f32(sy) + 0.5) * spacing);

  let center = read_pheromone(i32(sx), i32(sy), cols, rows, sampleCount);
  let right = read_pheromone(i32(sx) + 1, i32(sy), cols, rows, sampleCount);
  let left = read_pheromone(i32(sx) - 1, i32(sy), cols, rows, sampleCount);
  let down = read_pheromone(i32(sx), i32(sy) + 1, cols, rows, sampleCount);
  let up = read_pheromone(i32(sx), i32(sy) - 1, cols, rows, sampleCount);

  let diffusion = clamp(params.heatmapB.z, 0.01, 0.95);
  let trailDecay = max(params.heatmapB.w, 0.01);
  let fearDecay = max(params.lifeA.w, 0.01);
  let diffBlend = clamp(diffusion * dt * 6.0, 0.01, 0.48);

  let trailNeighborMean = (center.x + right.x + left.x + down.x + up.x) * 0.2;
  let fearNeighborMean = (center.y + right.y + left.y + down.y + up.y) * 0.2;
  var trail = mix(center.x, trailNeighborMean, diffBlend);
  var fear = mix(center.y, fearNeighborMean, diffBlend * 1.2);

  let boidSampleCount = min(boidCount, MAX_BOID_SAMPLES);
  let seed = hash_u32(idx * 1664525u + u32(params.counts.z) * 1013904223u + 23u);
  var boidIndex = seed % boidCount;
  var step = 1u;
  if (boidCount > 1u) {
    step = 1u + ((seed >> 1u) % (boidCount - 1u));
  }

  var trailDeposit = 0.0;
  for (var i: u32 = 0u; i < boidSampleCount; i = i + 1u) {
    let boid = boids[boidIndex];
    let dx = wrapped_abs_delta(samplePos.x, boid.posVel.x, worldW);
    let dy = wrapped_abs_delta(samplePos.y, boid.posVel.y, worldH);
    let distSq = dx * dx + dy * dy;
    if (distSq < diffuseRadiusSq) {
      trailDeposit = trailDeposit + exp(-distSq / max(diffuseRadiusSq * 0.62, 0.0001));
    }
    boidIndex = (boidIndex + step) % boidCount;
  }
  trailDeposit = trailDeposit * (f32(boidCount) / f32(boidSampleCount)) * 0.0022;

  var fearDeposit = 0.0;
  let fearRadiusSq = params.boidA.z * params.boidA.z;
  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    let predator = predators[p];
    let dx = wrapped_abs_delta(samplePos.x, predator.posVel.x, worldW);
    let dy = wrapped_abs_delta(samplePos.y, predator.posVel.y, worldH);
    let distSq = dx * dx + dy * dy;
    if (distSq < fearRadiusSq) {
      fearDeposit = fearDeposit + exp(-distSq / max(fearRadiusSq * 0.28, 0.0001));
    }
  }
  fearDeposit = fearDeposit * 0.055;

  trail = max(0.0, trail + trailDeposit - trail * trailDecay * dt);
  fear = max(0.0, fear + fearDeposit - fear * fearDecay * dt);

  pheromoneNext[idx].values = vec4f(trail, fear, trail - center.x, fear - center.y);
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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
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
struct HeatSample {
  densityDelta: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec3f,
  @location(2) alpha: f32,
};

@group(0) @binding(0) var<storage, read> heatSamples: array<HeatSample>;
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
  let worldW = params.world.x;
  let worldH = params.world.y;
  let dotSpacing = max(params.heatmapA.x, 0.75);
  let dotRadius = max(params.heatmapA.y, 0.5);

  let cols = max(1u, u32(floor(worldW / dotSpacing)));
  let sx = instanceIndex % cols;
  let sy = instanceIndex / cols;
  let center = vec2f((f32(sx) + 0.5) * dotSpacing, (f32(sy) + 0.5) * dotSpacing);
  let local = quad_vertex(vertexIndex);
  let world = center + local * dotRadius;

  let density = max(0.0, heatSamples[instanceIndex].densityDelta.x);
  let trendSignal = heatSamples[instanceIndex].densityDelta.y;
  let intensity = 1.0 - exp(-density * 0.09);
  let trend = clamp(trendSignal, -1.0, 1.0);
  let lift = smoothstep(0.0, 0.78, intensity);

  let dark = vec3f(0.03, 0.06, 0.17);
  let neutral = vec3f(0.62, 0.64, 0.68);
  let up = vec3f(1.00, 0.20, 0.14);
  let down = vec3f(0.18, 0.46, 1.00);
  let hue = select(mix(neutral, down, -trend), mix(neutral, up, trend), trend >= 0.0);
  let color = mix(dark, hue, 0.22 + 0.78 * lift);

  let clipX = world.x / worldW * 2.0 - 1.0;
  let clipY = 1.0 - world.y / worldH * 2.0;

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.local = local;
  out.color = color;
  out.alpha = 0.11 + 0.89 * smoothstep(0.0, 1.0, intensity);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let distSq = dot(in.local, in.local);
  if (distSq > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(0.65, 1.0, distSq);
  let alpha = in.alpha * (0.42 + 0.58 * edge);
  return vec4f(in.color * alpha, alpha);
}
`;

const heatmapSampleComputeShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
};

struct HeatSample {
  densityDelta: vec4f,
};

struct SimParams {
  world: vec4f,
  counts: vec4f,
  boidA: vec4f,
  boidB: vec4f,
  boidC: vec4f,
  predatorA: vec4f,
  predatorB: vec4f,
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<storage, read> heatPrev: array<HeatSample>;
@group(0) @binding(2) var<storage, read_write> heatNext: array<HeatSample>;
@group(0) @binding(3) var<uniform> params: SimParams;

const MIN_SAMPLES: u32 = 96u;
const MAX_SAMPLES: u32 = 768u;

fn wrapped_abs_delta(a: f32, b: f32, size: f32) -> f32 {
  let direct = abs(a - b);
  return min(direct, size - direct);
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

@compute @workgroup_size(${HEATMAP_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let sampleCount = u32(params.counts.w);
  if (idx >= sampleCount) {
    return;
  }

  let worldW = params.world.x;
  let worldH = params.world.y;
  let boidCount = max(1u, u32(params.counts.x));
  let dt = params.world.z;
  let dotSpacing = max(params.heatmapA.x, 0.75);
  let densityRadius = max(params.heatmapA.z, 2.0);
  let densityRadiusSq = densityRadius * densityRadius;
  let sampleBudget = clamp(u32(params.heatmapA.w), MIN_SAMPLES, MAX_SAMPLES);
  let trendGain = max(params.heatmapB.x, 0.01);
  let trendDeadband = max(params.heatmapB.y, 0.0);

  let cols = max(1u, u32(floor(worldW / dotSpacing)));
  let sx = idx % cols;
  let sy = idx / cols;
  let samplePos = vec2f((f32(sx) + 0.5) * dotSpacing, (f32(sy) + 0.5) * dotSpacing);

  let adaptiveSamples = clamp(boidCount / 48u + 96u, MIN_SAMPLES, sampleBudget);
  let sampleBoids = min(boidCount, adaptiveSamples);
  let frameSeed = u32(params.counts.z);
  let baseSeed = hash_u32(idx * 747796405u + frameSeed * 1664525u + 19u);
  var boidIndex = baseSeed % boidCount;
  var step = 1u;
  if (boidCount > 1u) {
    step = 1u + ((baseSeed >> 1u) % (boidCount - 1u));
  }

  var density = 0.0;
  for (var i: u32 = 0u; i < sampleBoids; i = i + 1u) {
    let boid = boids[boidIndex];
    let dx = wrapped_abs_delta(samplePos.x, boid.posVel.x, worldW);
    let dy = wrapped_abs_delta(samplePos.y, boid.posVel.y, worldH);
    let distSq = dx * dx + dy * dy;
    if (distSq < densityRadiusSq) {
      density = density + exp(-distSq / max(densityRadiusSq * 0.62, 0.0001));
    }
    boidIndex = (boidIndex + step) % boidCount;
  }

  let rawDensity = density * (f32(boidCount) / f32(sampleBoids));
  let previousDensity = heatPrev[idx].densityDelta.x;
  let previousTrend = heatPrev[idx].densityDelta.y;
  let previousFast = heatPrev[idx].densityDelta.z;
  let previousSlow = heatPrev[idx].densityDelta.w;

  let fastBlend = clamp(dt * 9.5, 0.12, 0.32);
  let slowBlend = clamp(dt * 2.1, 0.03, 0.12);
  let fastDensity = previousFast + (rawDensity - previousFast) * fastBlend;
  let slowDensity = previousSlow + (rawDensity - previousSlow) * slowBlend;
  let smoothedDensity = fastDensity + (previousDensity - fastDensity) * 0.08;

  let trendRaw = fastDensity - slowDensity;
  let trendMagnitude = max(0.0, abs(trendRaw) - trendDeadband);
  let trendSigned = sign(trendRaw) * trendMagnitude;
  let trendNormalized = clamp(trendSigned * trendGain, -1.0, 1.0);
  let smoothedTrend = previousTrend + (trendNormalized - previousTrend) * 0.18;

  heatNext[idx].densityDelta = vec4f(smoothedDensity, smoothedTrend, fastDensity, slowDensity);
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
  heatmapA: vec4f,
  heatmapB: vec4f,
  lifeA: vec4f,
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

function getHeatmapDotSpacingWorld() {
  return Math.max(0.75, config.heatmapDotSpacingPx / devicePixelRatioCached);
}

function getHeatmapDotRadiusWorld() {
  return Math.max(0.5, config.heatmapDotRadiusPx / devicePixelRatioCached);
}

function getHeatmapDiffuseRadiusWorld() {
  return Math.max(2, config.heatmapDiffuseRadiusPx / devicePixelRatioCached);
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
    boidNeighborSearch: 'uniform-grid',
    lifeFeatures: ['pheromone_trail', 'fear_field', 'panic_state'],
    coordinateSystem: 'origin top-left, +x right, +y down',
    viewport: { width: worldWidth, height: worldHeight },
    boidCount: BOID_COUNT,
    predatorCount: PREDATOR_COUNT,
    frameIndex,
    viewMode: renderMode,
    heatmapPoints: heatmapPointCount,
    heatmapDotSpacingPx: config.heatmapDotSpacingPx,
    heatmapDotRadiusPx: config.heatmapDotRadiusPx,
    heatmapDiffuseRadiusPx: config.heatmapDiffuseRadiusPx,
    heatmapSampleBudget: config.heatmapSampleBudget,
    heatmapTrendGain: config.heatmapTrendGain,
    heatmapTrendDeadband: config.heatmapTrendDeadband,
    pherTrailWeight: config.pherTrailWeight,
    pherFearWeight: config.pherFearWeight,
    pherDiffusion: config.pherDiffusion,
    pherDecay: config.pherDecay,
    panicBoost: config.panicBoost,
    gridCellSize: GRID_CELL_SIZE,
    gridMaxCells: GRID_MAX_CELLS,
    gridCellCapacity: GRID_CELL_CAPACITY,
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
  const dotSpacingWorld = getHeatmapDotSpacingWorld();
  const cols = Math.max(1, Math.floor(worldWidth / dotSpacingWorld));
  const rows = Math.max(1, Math.floor(worldHeight / dotSpacingWorld));
  heatmapPointCount = Math.min(cols * rows, HEATMAP_MAX_POINTS);

  canvas.width = Math.floor(worldWidth * dpr);
  canvas.height = Math.floor(worldHeight * dpr);
}

function createParamsArray(dtSeconds) {
  const frameScale = dtSeconds * 60;
  const predatorMaxTurnRate = config.maxTurnRate * config.predatorTurnRateFactor;
  const predatorMaxTurnAcceleration = config.maxTurnAcceleration * config.predatorTurnAccelerationFactor;
  return new Float32Array([
    worldWidth, worldHeight, dtSeconds, frameScale,
    BOID_COUNT, PREDATOR_COUNT, frameIndex, heatmapPointCount,
    config.perceptionRadius, config.separationRadius, config.predatorAvoidRadius, config.predatorCatchRadius,
    config.maxSpeed, config.minSpeed, config.maxForce, config.maxTurnRate,
    config.maxTurnAcceleration, config.alignWeight, config.cohesionWeight, config.separationWeight,
    config.predatorAvoidWeight, config.predatorAttentionSeconds, config.predatorSeparationRadius, config.predatorSeparationWeight,
    config.predatorPauseSlowdownSeconds, config.predatorSpeedFactor, predatorMaxTurnRate, predatorMaxTurnAcceleration,
    getHeatmapDotSpacingWorld(), getHeatmapDotRadiusWorld(), getHeatmapDiffuseRadiusWorld(), config.heatmapSampleBudget,
    config.heatmapTrendGain, config.heatmapTrendDeadband, config.pherDiffusion, config.pherDecay,
    config.pherTrailWeight, config.pherFearWeight, config.panicBoost, config.pherDecay * 1.35,
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
  const zeroGridCounts = new Uint32Array(GRID_MAX_CELLS);
  const zeroHeat = new Float32Array(HEATMAP_MAX_POINTS * 4);
  const zeroPheromone = new Float32Array(HEATMAP_MAX_POINTS * PHEROMONE_FLOATS);

  gpu.device.queue.writeBuffer(gpu.boidBuffers[0], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.boidBuffers[1], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.predatorBuffer, 0, predatorData);
  gpu.device.queue.writeBuffer(gpu.caughtFlagsBuffer, 0, zeroFlags);
  gpu.device.queue.writeBuffer(gpu.gridCellCountBuffer, 0, zeroGridCounts);
  gpu.device.queue.writeBuffer(gpu.heatmapBuffers[0], 0, zeroHeat);
  gpu.device.queue.writeBuffer(gpu.heatmapBuffers[1], 0, zeroHeat);
  gpu.device.queue.writeBuffer(gpu.pheromoneBuffers[0], 0, zeroPheromone);
  gpu.device.queue.writeBuffer(gpu.pheromoneBuffers[1], 0, zeroPheromone);

  frameIndex = 0;
  accumulator = 0;
  currentBoidBufferIndex = 0;
  currentHeatmapBufferIndex = 0;
  currentPheromoneBufferIndex = 0;
  writeParams(FIXED_STEP);
  updateTextState();
}

function stepSimulation(dtSeconds) {
  if (!gpu) {
    return { simCpuMs: 0, heatmapCpuMs: 0 };
  }
  const simStart = performance.now();
  let heatmapCpuMs = 0;

  const boidUpdateBindGroup = gpu.boidUpdateBindGroups[currentBoidBufferIndex][currentPheromoneBufferIndex];
  const predatorUpdateBindGroup = gpu.predatorUpdateBindGroups[currentBoidBufferIndex];
  const gridBuildBindGroup = gpu.gridBuildBindGroups[currentBoidBufferIndex];
  if (!boidUpdateBindGroup || !predatorUpdateBindGroup || !gridBuildBindGroup) {
    throw new Error(`Missing compute bind group for boid buffer index ${currentBoidBufferIndex}.`);
  }

  writeParams(dtSeconds);

  const encoder = gpu.device.createCommandEncoder();
  const simPass = encoder.beginComputePass();

  try {
    simPass.setPipeline(gpu.clearCaughtPipeline);
    simPass.setBindGroup(0, gpu.clearCaughtBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(PREDATOR_COUNT / CATCH_CLEAR_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.gridClearPipeline);
    simPass.setBindGroup(0, gpu.gridClearBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(GRID_MAX_CELLS / GRID_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.gridBuildPipeline);
    simPass.setBindGroup(0, gridBuildBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / BOID_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.predatorUpdatePipeline);
    simPass.setBindGroup(0, predatorUpdateBindGroup);
    simPass.dispatchWorkgroups(1);

    simPass.setPipeline(gpu.boidUpdatePipeline);
    simPass.setBindGroup(0, boidUpdateBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / BOID_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.predatorResolvePipeline);
    simPass.setBindGroup(0, gpu.predatorResolveBindGroup);
    simPass.dispatchWorkgroups(1);
  } finally {
    simPass.end();
  }

  currentBoidBufferIndex = 1 - currentBoidBufferIndex;
  frameIndex += 1;

  const pheromoneUpdateBindGroup = gpu.pheromoneUpdateBindGroups[currentBoidBufferIndex][currentPheromoneBufferIndex];
  if (!pheromoneUpdateBindGroup) {
    throw new Error('Missing pheromone compute bind group.');
  }

  const pheromoneStart = performance.now();
  const pheromonePass = encoder.beginComputePass();
  try {
    pheromonePass.setPipeline(gpu.pheromoneUpdatePipeline);
    pheromonePass.setBindGroup(0, pheromoneUpdateBindGroup);
    pheromonePass.dispatchWorkgroups(Math.ceil(heatmapPointCount / HEATMAP_WORKGROUP_SIZE));
  } finally {
    pheromonePass.end();
  }
  heatmapCpuMs += performance.now() - pheromoneStart;
  currentPheromoneBufferIndex = 1 - currentPheromoneBufferIndex;

  writeParams(dtSeconds);

  const heatmapBindGroup = gpu.heatmapComputeBindGroups[currentBoidBufferIndex][currentHeatmapBufferIndex];
  if (!heatmapBindGroup) {
    throw new Error('Missing heatmap compute bind group.');
  }

  const heatmapStart = performance.now();
  const heatPass = encoder.beginComputePass();
  try {
    heatPass.setPipeline(gpu.heatmapComputePipeline);
    heatPass.setBindGroup(0, heatmapBindGroup);
    heatPass.dispatchWorkgroups(Math.ceil(heatmapPointCount / HEATMAP_WORKGROUP_SIZE));
  } finally {
    heatPass.end();
  }
  heatmapCpuMs = performance.now() - heatmapStart;

  currentHeatmapBufferIndex = 1 - currentHeatmapBufferIndex;

  gpu.device.queue.submit([encoder.finish()]);
  return { simCpuMs: performance.now() - simStart, heatmapCpuMs };
}

function renderFrame() {
  if (!gpu) {
    return;
  }

  const boidRenderBindGroup = gpu.boidRenderBindGroups[currentBoidBufferIndex];
  const heatmapRenderBindGroup = gpu.boidHeatmapRenderBindGroups[currentHeatmapBufferIndex];
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
      renderPass.draw(6, heatmapPointCount, 0, 0);
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

function updatePerformancePanel() {
  if (!fpsValue || !frameCpuValue || !simCpuValue || !heatmapCpuValue || !renderCpuValue) {
    return;
  }
  fpsValue.textContent = perfStats.fps.toFixed(1);
  frameCpuValue.textContent = perfStats.frameCpuMs.toFixed(2);
  simCpuValue.textContent = perfStats.simCpuMs.toFixed(2);
  heatmapCpuValue.textContent = perfStats.heatmapCpuMs.toFixed(2);
  renderCpuValue.textContent = perfStats.renderCpuMs.toFixed(2);
}

function recordPerformance(frameCpuMs, simCpuMs, heatmapCpuMs, renderCpuMs, instantFps, timestamp) {
  const alpha = perfStats.alpha;
  perfStats.fps = perfStats.fps + (instantFps - perfStats.fps) * alpha;
  perfStats.frameCpuMs = perfStats.frameCpuMs + (frameCpuMs - perfStats.frameCpuMs) * alpha;
  perfStats.simCpuMs = perfStats.simCpuMs + (simCpuMs - perfStats.simCpuMs) * alpha;
  perfStats.heatmapCpuMs = perfStats.heatmapCpuMs + (heatmapCpuMs - perfStats.heatmapCpuMs) * alpha;
  perfStats.renderCpuMs = perfStats.renderCpuMs + (renderCpuMs - perfStats.renderCpuMs) * alpha;

  if (timestamp - perfStats.lastPanelUpdate >= 180) {
    perfStats.lastPanelUpdate = timestamp;
    updatePerformancePanel();
  }
}

function tick(timestamp) {
  if (!gpu || gpuRuntimeError) {
    return;
  }

  try {
    const frameCpuStart = performance.now();
    if (lastFrameTime === 0) {
      lastFrameTime = timestamp;
    }

    const rawDelta = (timestamp - lastFrameTime) / 1000;
    const instantFps = rawDelta > 0.000001 ? 1 / rawDelta : 0;
    lastFrameTime = timestamp;
    accumulator += Math.min(rawDelta, 0.05);
    let simCpuMs = 0;
    let heatmapCpuMs = 0;

    while (accumulator >= FIXED_STEP) {
      const timings = stepSimulation(FIXED_STEP);
      simCpuMs += timings.simCpuMs;
      heatmapCpuMs += timings.heatmapCpuMs;
      accumulator -= FIXED_STEP;
    }

    const renderStart = performance.now();
    renderFrame();
    const renderCpuMs = performance.now() - renderStart;
    const frameCpuMs = performance.now() - frameCpuStart;
    recordPerformance(frameCpuMs, simCpuMs, heatmapCpuMs, renderCpuMs, instantFps, timestamp);
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
  const heatmapBufferBytes = HEATMAP_MAX_POINTS * 4 * Float32Array.BYTES_PER_ELEMENT;
  const pheromoneBufferBytes = HEATMAP_MAX_POINTS * PHEROMONE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const gridCellCountBytes = GRID_MAX_CELLS * Uint32Array.BYTES_PER_ELEMENT;
  const gridBoidIndexBytes = GRID_MAX_CELLS * GRID_CELL_CAPACITY * Uint32Array.BYTES_PER_ELEMENT;
  const paramsBufferBytes = 40 * Float32Array.BYTES_PER_ELEMENT;
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

  const heatmapBuffers = [
    device.createBuffer({
      size: heatmapBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      size: heatmapBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];

  const pheromoneBuffers = [
    device.createBuffer({
      size: pheromoneBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      size: pheromoneBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];

  const gridCellCountBuffer = device.createBuffer({
    size: gridCellCountBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const gridBoidIndexBuffer = device.createBuffer({
    size: gridBoidIndexBytes,
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
  const gridClearModule = await createCheckedShaderModule('gridClearShader', gridClearShader);
  const gridBuildModule = await createCheckedShaderModule('gridBuildShader', gridBuildShader);
  const pheromoneUpdateModule = await createCheckedShaderModule('pheromoneUpdateShader', pheromoneUpdateShader);
  const clearCaughtModule = await createCheckedShaderModule('catchClearShader', catchClearShader);
  const predatorResolveModule = await createCheckedShaderModule('predatorResolveShader', predatorResolveShader);
  const boidRenderModule = await createCheckedShaderModule('boidRenderShader', boidRenderShader);
  const heatmapSampleComputeModule = await createCheckedShaderModule('heatmapSampleComputeShader', heatmapSampleComputeShader);
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

  const gridClearPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gridClearModule,
      entryPoint: 'main',
    },
  });

  const gridBuildPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gridBuildModule,
      entryPoint: 'main',
    },
  });

  const pheromoneUpdatePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: pheromoneUpdateModule,
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

  const heatmapComputePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: heatmapSampleComputeModule,
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
	              dstFactor: 'one-minus-src-alpha',
	              operation: 'add',
	            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
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
    [
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: boidBuffers[1] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: gridCellCountBuffer } },
          { binding: 5, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 6, resource: { buffer: caughtFlagsBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: boidBuffers[1] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: gridCellCountBuffer } },
          { binding: 5, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 6, resource: { buffer: caughtFlagsBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
    [
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: boidBuffers[0] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: gridCellCountBuffer } },
          { binding: 5, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 6, resource: { buffer: caughtFlagsBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: boidBuffers[0] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: gridCellCountBuffer } },
          { binding: 5, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 6, resource: { buffer: caughtFlagsBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
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

  const gridClearBindGroup = device.createBindGroup({
    layout: gridClearPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gridCellCountBuffer } }],
  });

  const gridBuildBindGroups = [
    device.createBindGroup({
      layout: gridBuildPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[0] } },
        { binding: 1, resource: { buffer: gridCellCountBuffer } },
        { binding: 2, resource: { buffer: gridBoidIndexBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: gridBuildPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boidBuffers[1] } },
        { binding: 1, resource: { buffer: gridCellCountBuffer } },
        { binding: 2, resource: { buffer: gridBoidIndexBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    }),
  ];

  const pheromoneUpdateBindGroups = [
    [
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
    [
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
  ];

  const heatmapComputeBindGroups = [
    [
      device.createBindGroup({
        layout: heatmapComputePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: heatmapBuffers[0] } },
          { binding: 2, resource: { buffer: heatmapBuffers[1] } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: heatmapComputePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: heatmapBuffers[1] } },
          { binding: 2, resource: { buffer: heatmapBuffers[0] } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
    [
      device.createBindGroup({
        layout: heatmapComputePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: heatmapBuffers[0] } },
          { binding: 2, resource: { buffer: heatmapBuffers[1] } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: heatmapComputePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: heatmapBuffers[1] } },
          { binding: 2, resource: { buffer: heatmapBuffers[0] } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      }),
    ],
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
        { binding: 0, resource: { buffer: heatmapBuffers[0] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: boidHeatmapRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: heatmapBuffers[1] } },
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
    heatmapBuffers,
    pheromoneBuffers,
    gridCellCountBuffer,
    gridBoidIndexBuffer,
    caughtFlagsBuffer,
    paramsBuffer,
    boidUpdatePipeline,
    predatorUpdatePipeline,
    gridClearPipeline,
    gridBuildPipeline,
    pheromoneUpdatePipeline,
    clearCaughtPipeline,
    predatorResolvePipeline,
    heatmapComputePipeline,
    boidRenderPipeline,
    boidHeatmapRenderPipeline,
    predatorRenderPipeline,
    boidUpdateBindGroups,
    predatorUpdateBindGroups,
    gridClearBindGroup,
    gridBuildBindGroups,
    pheromoneUpdateBindGroups,
    heatmapComputeBindGroups,
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

function updateHeatmapControlLabels() {
  heatmapSpacingValue.textContent = config.heatmapDotSpacingPx.toFixed(1);
  heatmapDotRadiusValue.textContent = config.heatmapDotRadiusPx.toFixed(2);
  heatmapDiffuseValue.textContent = config.heatmapDiffuseRadiusPx.toFixed(0);
  heatmapSamplesValue.textContent = config.heatmapSampleBudget.toFixed(0);
  heatmapTrendGainValue.textContent = config.heatmapTrendGain.toFixed(1);
  heatmapTrendDeadbandValue.textContent = config.heatmapTrendDeadband.toFixed(3);
}

function updateLifeFieldControlLabels() {
  pherTrailWeightValue.textContent = config.pherTrailWeight.toFixed(2);
  pherFearWeightValue.textContent = config.pherFearWeight.toFixed(2);
  pherDiffusionValue.textContent = config.pherDiffusion.toFixed(2);
  pherDecayValue.textContent = config.pherDecay.toFixed(2);
  panicBoostValue.textContent = config.panicBoost.toFixed(2);
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

heatmapSpacingRange.addEventListener('input', () => {
  config.heatmapDotSpacingPx = Number(heatmapSpacingRange.value);
  updateHeatmapControlLabels();
  handleResize();
});

heatmapDotRadiusRange.addEventListener('input', () => {
  config.heatmapDotRadiusPx = Number(heatmapDotRadiusRange.value);
  updateHeatmapControlLabels();
});

heatmapDiffuseRange.addEventListener('input', () => {
  config.heatmapDiffuseRadiusPx = Number(heatmapDiffuseRange.value);
  updateHeatmapControlLabels();
});

heatmapSamplesRange.addEventListener('input', () => {
  config.heatmapSampleBudget = Number(heatmapSamplesRange.value);
  updateHeatmapControlLabels();
});

heatmapTrendGainRange.addEventListener('input', () => {
  config.heatmapTrendGain = Number(heatmapTrendGainRange.value);
  updateHeatmapControlLabels();
});

heatmapTrendDeadbandRange.addEventListener('input', () => {
  config.heatmapTrendDeadband = Number(heatmapTrendDeadbandRange.value);
  updateHeatmapControlLabels();
});

pherTrailWeightRange.addEventListener('input', () => {
  config.pherTrailWeight = Number(pherTrailWeightRange.value);
  updateLifeFieldControlLabels();
});

pherFearWeightRange.addEventListener('input', () => {
  config.pherFearWeight = Number(pherFearWeightRange.value);
  updateLifeFieldControlLabels();
});

pherDiffusionRange.addEventListener('input', () => {
  config.pherDiffusion = Number(pherDiffusionRange.value);
  updateLifeFieldControlLabels();
});

pherDecayRange.addEventListener('input', () => {
  config.pherDecay = Number(pherDecayRange.value);
  updateLifeFieldControlLabels();
});

panicBoostRange.addEventListener('input', () => {
  config.panicBoost = Number(panicBoostRange.value);
  updateLifeFieldControlLabels();
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
updateHeatmapControlLabels();
updateLifeFieldControlLabels();
updateViewToggleLabel();
updatePerformancePanel();

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
