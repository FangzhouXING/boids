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
const metabolismRange = document.getElementById('metabolismRange');
const metabolismValue = document.getElementById('metabolismValue');
const boidSprintBoostRange = document.getElementById('boidSprintBoostRange');
const boidSprintBoostValue = document.getElementById('boidSprintBoostValue');
const predatorSprintBoostRange = document.getElementById('predatorSprintBoostRange');
const predatorSprintBoostValue = document.getElementById('predatorSprintBoostValue');
const confusionStrengthRange = document.getElementById('confusionStrengthRange');
const confusionStrengthValue = document.getElementById('confusionStrengthValue');
const mutationRateRange = document.getElementById('mutationRateRange');
const mutationRateValue = document.getElementById('mutationRateValue');
const fpsValue = document.getElementById('fpsValue');
const frameCpuValue = document.getElementById('frameCpuValue');
const simCpuValue = document.getElementById('simCpuValue');
const heatmapCpuValue = document.getElementById('heatmapCpuValue');
const renderCpuValue = document.getElementById('renderCpuValue');

const FIXED_STEP = 1 / 60;
const BOID_COUNT = 30000;
const PREDATOR_COUNT = 1;
const BOID_WORKGROUP_SIZE = 128;
const CATCH_CLEAR_WORKGROUP_SIZE = 32;
const HEATMAP_WORKGROUP_SIZE = 128;
const GRID_WORKGROUP_SIZE = 128;
const BOID_FLOATS = 12;
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
const SIM_COUNTER_COUNT = 2;
const BOID_STATS_SAMPLE_COUNT = 512;
const ECOLOGY_READBACK_INTERVAL_MS = 260;
const PHYS_SENSOR_ANGLE = (28 * Math.PI) / 180;
const PHYS_SENSOR_OFFSET = 10.5;
const PHYS_SENSOR_WIDTH = 1;
const PHYS_ROTATE_ANGLE = (26 * Math.PI) / 180;
const PHYS_STEP_SIZE = 1;
const PHYS_DEPOSIT = 0.65;
const PHYS_DIFFUSE_MIX = 0.36;
const PHYS_DECAY = 0.05;
const PHYS_MODE_REPEL = 0;
const PHYS_NODE_SOURCE_STRENGTH = 0.08;

const config = {
  perceptionRadius: 92,
  separationRadius: 18,
  maxSpeed: 1.7,
  minSpeed: Number(minSpeedRange.value),
  maxForce: 0.04,
  maxTurnRate: (Number(panicBoostRange?.value || 145) * Math.PI) / 180,
  maxTurnAcceleration: (Number(turnAccelRange.value) * Math.PI) / 180,
  alignWeight: 0.34,
  cohesionWeight: 0.07,
  separationWeight: 0.52,
  predatorAvoidWeight: 2.95,
  predatorCatchRadius: 9,
  predatorAvoidRadius: 132,
  predatorFearStrength: 0.18,
  predatorFearRadius: 72,
  predatorSeparationRadius: 50,
  predatorSeparationWeight: 1.8,
  predatorTurnRateFactor: 0.78,
  predatorTurnAccelerationFactor: 0.72,
  predatorSpeedFactor: 0.9,
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
  metabolismIntensity: Number(metabolismRange?.value || 1.0),
  boidSprintBoost: Number(boidSprintBoostRange?.value || 0.9),
  predatorSprintBoost: Number(predatorSprintBoostRange?.value || 0.2),
  confusionStrength: Number(confusionStrengthRange?.value || 0.62),
  mutationRate: Number(mutationRateRange?.value || 0.07),
  physSensorAngle: PHYS_SENSOR_ANGLE,
  physSensorOffset: Number(pherFearWeightRange?.value || 10.5),
  physSensorWidth: PHYS_SENSOR_WIDTH,
  physRotateAngle: PHYS_ROTATE_ANGLE,
  physStepSize: PHYS_STEP_SIZE,
  physDeposit: Number(pherTrailWeightRange?.value || 0.65),
  physDiffuseMix: Number(pherDiffusionRange?.value || 0.36),
  physDecay: Number(pherDecayRange?.value || 0.05),
  physModeRepel: PHYS_MODE_REPEL,
  physNodeSourceStrength: Number(predatorAttentionRange?.value || 0.08),
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
const ecologyStats = {
  avgSpeed: Math.max(0.2, Math.min(1.6, config.minSpeed * 1.25)),
  avgPredatorSpeed: 0,
  boidsNearPredatorsFraction: 0,
  movedFraction: 0,
  movesPerSecond: 0,
  blockedMoveRate: 0,
  predatorSamples: [],
  lastReadbackAtMs: 0,
  lastCounterSampleAtMs: 0,
  readbackPending: false,
};

const boidUpdateShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
  bio: vec4f,
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
  lifeB: vec4f,
};

@group(0) @binding(0) var<storage, read> boidsIn: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> boidsOut: array<BoidState>;
@group(0) @binding(2) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(3) var<storage, read> pheromones: array<PheromoneSample>;
@group(0) @binding(4) var<storage, read_write> occupancyClaims: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> cellBoids: array<u32>;
@group(0) @binding(7) var<uniform> params: SimParams;
@group(0) @binding(8) var<storage, read_write> simCounters: array<atomic<u32>>;

const TAU: f32 = 6.28318530718;
const PHEROMONE_MAX_POINTS: u32 = ${HEATMAP_MAX_POINTS}u;
const GRID_CELL_SIZE: f32 = ${GRID_CELL_SIZE.toFixed(1)};
const GRID_MAX_CELLS: u32 = ${GRID_MAX_CELLS}u;
const GRID_CELL_CAPACITY: u32 = ${GRID_CELL_CAPACITY}u;
const MAX_CELL_SCAN: u32 = 24u;
const COUNTER_MOVED: u32 = 0u;
const COUNTER_BLOCKED: u32 = 1u;

fn clamp_coordinate(value: f32, size: f32) -> f32 {
  return clamp(value, 0.0, max(size - 0.001, 0.0));
}

fn read_pheromone_sample(x: i32, y: i32, cols: u32, rows: u32, sampleCount: u32) -> vec4f {
  if (cols == 0u || rows == 0u || sampleCount == 0u) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  if (x < 0 || y < 0 || x >= i32(cols) || y >= i32(rows)) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  let idx = u32(y) * cols + u32(x);
  if (idx >= sampleCount || idx >= PHEROMONE_MAX_POINTS) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  return pheromones[idx].values;
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

fn safe_normalize(v: vec2f) -> vec2f {
  let len = length(v);
  if (len <= 0.00001) {
    return vec2f(0.0, 0.0);
  }
  return v / len;
}

fn limit_magnitude(v: vec2f, maxLen: f32) -> vec2f {
  let len = length(v);
  if (len <= maxLen || len <= 0.00001) {
    return v;
  }
  return v * (maxLen / len);
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

fn sensor_mean_component(
  sensorPos: vec2f,
  radius: i32,
  cols: u32,
  rows: u32,
  sampleCount: u32,
  spacing: f32,
  component: u32,
) -> f32 {
  if (sampleCount == 0u || spacing <= 0.0) {
    return 0.0;
  }
  let baseX = i32(floor(sensorPos.x / spacing));
  let baseY = i32(floor(sensorPos.y / spacing));
  var sum = 0.0;
  var count = 0.0;
  for (var oy: i32 = -radius; oy <= radius; oy = oy + 1) {
    for (var ox: i32 = -radius; ox <= radius; ox = ox + 1) {
      let sample = read_pheromone_sample(baseX + ox, baseY + oy, cols, rows, sampleCount);
      var sampleValue = sample.x;
      if (component == 1u) {
        sampleValue = sample.y;
      } else if (component == 2u) {
        sampleValue = sample.z;
      } else if (component == 3u) {
        sampleValue = sample.w;
      }
      sum = sum + sampleValue;
      count = count + 1.0;
    }
  }
  return sum / max(count, 1.0);
}

@compute @workgroup_size(${BOID_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let boidCount = u32(params.counts.x);
  let index = gid.x;
  if (index >= boidCount) {
    return;
  }

  let worldW = params.world.x;
  let worldH = params.world.y;
  let dt = max(params.world.z, 0.0001);
  let frameScale = params.world.w;

  let predatorCount = u32(params.counts.y);
  let pherSpacing = max(params.heatmapA.x, 0.75);
  let pherCols = max(1u, u32(floor(worldW / pherSpacing)));
  let pherRows = max(1u, u32(floor(worldH / pherSpacing)));
  let pherCount = min(PHEROMONE_MAX_POINTS, pherCols * pherRows);

  let gridCols = max(1u, u32(ceil(worldW / GRID_CELL_SIZE)));
  let gridRows = max(1u, u32(ceil(worldH / GRID_CELL_SIZE)));
  let gridCount = min(GRID_MAX_CELLS, gridCols * gridRows);

  let sensorAngle = params.boidA.x;
  let sensorOffset = params.boidA.y;
  let sensorWidth = max(1.0, params.boidA.z);
  let cruiseSpeed = max(0.05, params.boidA.w);
  let maxTurnRate = max(0.05, params.boidB.x);
  let alignmentWeight = max(0.0, params.boidB.y);
  let maxTurnAcceleration = max(0.05, params.boidC.x);
  let cohesionWeight = max(0.0, params.boidC.y);
  let separationWeight = max(0.0, params.boidC.z);
  let trailWeight = max(0.0, params.boidC.w);
  let perceptionRadius = max(10.0, params.predatorA.z);
  let separationRadius = max(5.0, params.predatorA.w);
  let predatorAvoidWeight = max(0.0, params.predatorA.x);
  let predatorAvoidRadius = max(8.0, params.lifeA.z);
  let predatorAvoidRadiusSq = predatorAvoidRadius * predatorAvoidRadius;
  let fearAvoidWeight = max(0.0, params.lifeA.w);
  let perceptionRadiusSq = perceptionRadius * perceptionRadius;
  let separationRadiusSq = separationRadius * separationRadius;
  let sensorRadius = i32(max(0.0, floor(sensorWidth * 0.5)));

  var me = boidsIn[index];
  let mePos = me.posVel.xy;
  let heading = me.headingTurn.x;
  let currentTurnRate = me.headingTurn.w;

  var predatorAvoidVec = vec2f(0.0, 0.0);
  var predatorThreat = 0.0;
  var predatorAvoidCount = 0u;
  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    let pred = predators[p];
    let dx = pred.posVel.x - mePos.x;
    let dy = pred.posVel.y - mePos.y;
    let delta = vec2f(dx, dy);
    let distSq = dot(delta, delta);
    if (distSq <= 0.0001 || distSq > predatorAvoidRadiusSq) {
      continue;
    }
    let dist = sqrt(distSq);
    let proximity = 1.0 - dist / predatorAvoidRadius;
    predatorThreat = max(predatorThreat, proximity);
    predatorAvoidVec = predatorAvoidVec - (delta / max(dist, 0.0001)) * (0.35 + 0.65 * proximity);
    predatorAvoidCount = predatorAvoidCount + 1u;
  }
  if (predatorAvoidCount > 0u) {
    predatorAvoidVec = safe_normalize(predatorAvoidVec / f32(predatorAvoidCount));
  }

  let cx = min(gridCols - 1u, u32(floor(mePos.x / GRID_CELL_SIZE)));
  let cy = min(gridRows - 1u, u32(floor(mePos.y / GRID_CELL_SIZE)));
  var neighborCount: u32 = 0u;
  var closeCount: u32 = 0u;
  var cohesionOffsetSum = vec2f(0.0, 0.0);
  var separationSum = vec2f(0.0, 0.0);
  var alignmentSum = vec2f(0.0, 0.0);
  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    let nyi = i32(cy) + oy;
    if (nyi < 0 || nyi >= i32(gridRows)) {
      continue;
    }
    let ny = u32(nyi);
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let nxi = i32(cx) + ox;
      if (nxi < 0 || nxi >= i32(gridCols)) {
        continue;
      }
      let nx = u32(nxi);
      let rawCellIndex = ny * gridCols + nx;
      if (rawCellIndex >= gridCount) {
        continue;
      }
      let cellCount = min(atomicLoad(&cellCounts[rawCellIndex]), min(MAX_CELL_SCAN, GRID_CELL_CAPACITY));
      let baseOffset = rawCellIndex * GRID_CELL_CAPACITY;
      for (var slot: u32 = 0u; slot < cellCount; slot = slot + 1u) {
        let neighborIndex = cellBoids[baseOffset + slot];
        if (neighborIndex == index || neighborIndex >= boidCount) {
          continue;
        }
        let other = boidsIn[neighborIndex];
        let dx = other.posVel.x - mePos.x;
        let dy = other.posVel.y - mePos.y;
        let delta = vec2f(dx, dy);
        let distSq = dot(delta, delta);
        if (distSq <= 0.0001 || distSq > perceptionRadiusSq) {
          continue;
        }
        neighborCount = neighborCount + 1u;
        cohesionOffsetSum = cohesionOffsetSum + delta;
        alignmentSum = alignmentSum + safe_normalize(other.posVel.zw);
        if (distSq < separationRadiusSq) {
          let dist = sqrt(distSq);
          let away = -delta / max(dist, 0.0001);
          let weight = 1.0 - dist / separationRadius;
          separationSum = separationSum + away * weight;
          closeCount = closeCount + 1u;
        }
      }
    }
  }

  var cohesionVec = vec2f(0.0, 0.0);
  var alignmentVec = vec2f(0.0, 0.0);
  if (neighborCount > 0u) {
    cohesionVec = safe_normalize(cohesionOffsetSum / f32(neighborCount));
    alignmentVec = safe_normalize(alignmentSum / f32(neighborCount));
  }
  var separationVec = vec2f(0.0, 0.0);
  if (closeCount > 0u) {
    separationVec = separationSum / f32(closeCount);
  }
  separationVec = limit_magnitude(separationVec, 1.0);

  let dirF = vec2f(cos(heading), sin(heading));
  let dirL = vec2f(cos(heading + sensorAngle), sin(heading + sensorAngle));
  let dirR = vec2f(cos(heading - sensorAngle), sin(heading - sensorAngle));
  let sensorPosF = vec2f(
    clamp_coordinate(mePos.x + dirF.x * sensorOffset, worldW),
    clamp_coordinate(mePos.y + dirF.y * sensorOffset, worldH),
  );
  let sensorPosL = vec2f(
    clamp_coordinate(mePos.x + dirL.x * sensorOffset, worldW),
    clamp_coordinate(mePos.y + dirL.y * sensorOffset, worldH),
  );
  let sensorPosR = vec2f(
    clamp_coordinate(mePos.x + dirR.x * sensorOffset, worldW),
    clamp_coordinate(mePos.y + dirR.y * sensorOffset, worldH),
  );
  let F = sensor_mean_component(sensorPosF, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 0u);
  let FL = sensor_mean_component(sensorPosL, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 0u);
  let FR = sensor_mean_component(sensorPosR, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 0u);
  let fearF = sensor_mean_component(sensorPosF, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 1u);
  let fearL = sensor_mean_component(sensorPosL, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 1u);
  let fearR = sensor_mean_component(sensorPosR, sensorRadius, pherCols, pherRows, pherCount, pherSpacing, 1u);
  let fearVec = safe_normalize(dirF * fearF + dirL * fearL + dirR * fearR);
  let fearLevel = max(0.0, (fearF + fearL + fearR) * 0.33333334);
  let fearThreat = clamp(fearLevel * 1.8, 0.0, 1.0);
  let meCellX = i32(floor(mePos.x / pherSpacing));
  let meCellY = i32(floor(mePos.y / pherSpacing));
  let trailLeft = read_pheromone_sample(meCellX - 1, meCellY, pherCols, pherRows, pherCount).x;
  let trailRight = read_pheromone_sample(meCellX + 1, meCellY, pherCols, pherRows, pherCount).x;
  let trailUp = read_pheromone_sample(meCellX, meCellY - 1, pherCols, pherRows, pherCount).x;
  let trailDown = read_pheromone_sample(meCellX, meCellY + 1, pherCols, pherRows, pherCount).x;
  let antiTrailGradient = safe_normalize(vec2f(trailLeft - trailRight, trailUp - trailDown));
  let trailVec = safe_normalize(dirF * F + dirL * FL + dirR * FR);
  let trailLevel = max(0.0, (F + FL + FR) * 0.33333334);
  let trailSaturation = smoothstep(0.85, 2.8, trailLevel);
  let lowTrail = 1.0 - smoothstep(0.08, 0.65, trailLevel);
  let congestion = smoothstep(0.42, 1.6, trailLevel);
  let sideImbalance = clamp((FL - FR) / max(F + FL + FR, 0.0001), -1.0, 1.0);
  let lateral = vec2f(-dirF.y, dirF.x);
  let trailSteerVec = safe_normalize(trailVec + lateral * sideImbalance * 0.75);
  let forward = dirF;
  let seed = u32(params.counts.z) * 1664525u + index * 1013904223u + 23u;
  let jitterAngle = (rand01(seed) - 0.5) * 0.45;
  let jitterVec = vec2f(cos(heading + jitterAngle), sin(heading + jitterAngle));
  let threat = max(predatorThreat, fearThreat * 0.92);
  let trailInfluence = trailWeight / (1.0 + trailLevel * 0.7) * (1.0 - 0.84 * threat);
  let wallMargin = max(20.0, min(worldW, worldH) * 0.08);
  var wallAvoid = vec2f(0.0, 0.0);
  if (mePos.x < wallMargin) {
    wallAvoid.x = wallAvoid.x + (1.0 - mePos.x / wallMargin);
  } else if (mePos.x > worldW - wallMargin) {
    wallAvoid.x = wallAvoid.x - (1.0 - (worldW - mePos.x) / wallMargin);
  }
  if (mePos.y < wallMargin) {
    wallAvoid.y = wallAvoid.y + (1.0 - mePos.y / wallMargin);
  } else if (mePos.y > worldH - wallMargin) {
    wallAvoid.y = wallAvoid.y - (1.0 - (worldH - mePos.y) / wallMargin);
  }

  var desiredDir = forward * 0.28;
  desiredDir = desiredDir + alignmentVec * alignmentWeight;
  desiredDir = desiredDir + cohesionVec * cohesionWeight;
  desiredDir = desiredDir + separationVec * separationWeight;
  desiredDir = desiredDir + predatorAvoidVec * predatorAvoidWeight * (0.55 + threat * 1.25);
  desiredDir = desiredDir - fearVec * fearAvoidWeight * (0.35 + fearThreat * 1.25);
  desiredDir = desiredDir + trailSteerVec * trailInfluence;
  desiredDir = desiredDir - trailSteerVec * trailSaturation * 0.56;
  desiredDir = desiredDir + antiTrailGradient * congestion * 0.95;
  desiredDir = desiredDir + lateral * sideImbalance * congestion * 0.35;
  desiredDir = desiredDir + forward * trailSaturation * 0.14;
  desiredDir = desiredDir + jitterVec * (0.04 + lowTrail * 0.14 + 0.08 * threat);
  desiredDir = desiredDir + wallAvoid * (0.9 + 0.7 * threat);
  if (length(desiredDir) <= 0.00001) {
    desiredDir = forward;
  }
  let desiredHeading = atan2(desiredDir.y, desiredDir.x);
  let headingDelta = shortest_angle_delta(heading, desiredHeading);
  let desiredTurnRate = clamp(headingDelta / dt, -maxTurnRate, maxTurnRate);
  let turnRateDeltaLimit = maxTurnAcceleration * dt;
  var nextTurnRate = currentTurnRate + clamp(
    desiredTurnRate - currentTurnRate,
    -turnRateDeltaLimit,
    turnRateDeltaLimit,
  );
  nextTurnRate = clamp(nextTurnRate, -maxTurnRate, maxTurnRate);
  nextTurnRate = nextTurnRate * 0.9;
  let nextHeading = normalize_angle(heading + nextTurnRate * dt);

  let crowdFactor = clamp(f32(closeCount) / 6.0, 0.0, 1.0);
  let nextSpeed = cruiseSpeed * mix(1.0, 0.78, crowdFactor) * mix(1.0, 1.22, threat);
  let moveDir = vec2f(cos(nextHeading), sin(nextHeading));
  var nextVel = moveDir * nextSpeed;
  var nextPos = mePos + nextVel * frameScale;
  var boundedHeading = nextHeading;
  var boundedTurnRate = nextTurnRate;
  if (nextPos.x <= 0.0 || nextPos.x >= worldW) {
    nextVel.x = -nextVel.x;
    boundedHeading = atan2(nextVel.y, nextVel.x);
    boundedTurnRate = 0.0;
  }
  if (nextPos.y <= 0.0 || nextPos.y >= worldH) {
    nextVel.y = -nextVel.y;
    boundedHeading = atan2(nextVel.y, nextVel.x);
    boundedTurnRate = 0.0;
  }
  nextPos = vec2f(
    clamp_coordinate(nextPos.x, worldW),
    clamp_coordinate(nextPos.y, worldH),
  );

  let targetCellX = min(pherCols - 1u, u32(floor(nextPos.x / pherSpacing)));
  let targetCellY = min(pherRows - 1u, u32(floor(nextPos.y / pherSpacing)));
  let targetIndex = targetCellY * pherCols + targetCellX;
  if (targetIndex < pherCount) {
    atomicAdd(&occupancyClaims[targetIndex], 1u);
  }

  atomicAdd(&simCounters[COUNTER_MOVED], 1u);
  if (closeCount > 2u) {
    atomicAdd(&simCounters[COUNTER_BLOCKED], 1u);
  }

  boidsOut[index] = BoidState(
    vec4f(nextPos, nextVel),
    vec4f(boundedHeading, 1.0, 1.0 - crowdFactor * 0.5, boundedTurnRate),
    me.bio,
  );
}
`;

const predatorUpdateShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
  bio: vec4f,
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
  lifeB: vec4f,
};

@group(0) @binding(0) var<storage, read> boidsIn: array<BoidState>;
@group(0) @binding(1) var<storage, read_write> predators: array<PredatorState>;
@group(0) @binding(2) var<uniform> params: SimParams;

const MAX_PREDATORS: u32 = 16u;

fn clamp_coordinate(value: f32, size: f32) -> f32 {
  return clamp(value, 0.0, max(size - 0.001, 0.0));
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

  let predatorSprintBoost = max(params.predatorB.x, 0.0);
  let predatorCruiseSpeed = params.boidA.w * params.predatorB.y;
  let predatorMaxTurnRate = params.predatorB.z;
  let predatorMaxTurnAcceleration = params.predatorB.w;

  var snapshot: array<PredatorState, MAX_PREDATORS>;

  for (var i: u32 = 0u; i < predatorCount; i = i + 1u) {
    snapshot[i] = predators[i];
  }

  for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
    var pred = snapshot[p];
    let isApex = pred.aux.y > 0.5;
    pred.headingTimers.w = max(0.0, pred.headingTimers.w - dt);
    pred.aux.x = max(0.0, pred.aux.x - dt);
    pred.headingTimers.y = pred.headingTimers.y - dt;

    var targetIndex: i32 = i32(round(pred.aux.z));
    if (pred.headingTimers.y <= 0.0 || targetIndex < 0 || targetIndex >= i32(boidCount)) {
      var bestDistSq = 1e20;
      var closest: u32 = 0u;
      for (var b: u32 = 0u; b < boidCount; b = b + 1u) {
        let boid = boidsIn[b];
        let boidSpecies = boid.bio.w;
        let prefersForager = boidSpecies < 0.5;
        let preference = select(
          select(0.86, 1.22, prefersForager),
          select(1.18, 0.82, prefersForager),
          isApex,
        );
        let dx = boid.posVel.x - pred.posVel.x;
        let dy = boid.posVel.y - pred.posVel.y;
        let distSq = dx * dx + dy * dy;
        let weighted = distSq / max(preference, 0.2);
        if (weighted < bestDistSq) {
          bestDistSq = weighted;
          closest = b;
        }
      }
      targetIndex = i32(closest);
      pred.headingTimers.y = attentionSeconds * select(1.0, 1.3, isApex);
    }

    var chaseDir = vec2f(cos(pred.headingTimers.x), sin(pred.headingTimers.x));
    var targetDistance = 1e9;
    if (targetIndex >= 0 && targetIndex < i32(boidCount)) {
      let prey = boidsIn[u32(targetIndex)];
      let tx = prey.posVel.x - pred.posVel.x;
      let ty = prey.posVel.y - pred.posVel.y;
      let toPrey = vec2f(tx, ty);
      let tLen = length(toPrey);
      targetDistance = tLen;
      if (tLen > 0.0001) {
        let leadTime = clamp(tLen / max(predatorCruiseSpeed * 60.0, 0.0001) * 0.75, 0.0, 0.55);
        let led = toPrey + prey.posVel.zw * (leadTime * 60.0);
        if (length(led) > 0.0001) {
          chaseDir = normalize(led);
        } else {
          chaseDir = toPrey / tLen;
        }
      }
    }

    var separate = vec2f(0.0, 0.0);
    var separateCount = 0u;
    for (var q: u32 = 0u; q < predatorCount; q = q + 1u) {
      if (q == p) {
        continue;
      }
      let other = snapshot[q];
      let dx = other.posVel.x - pred.posVel.x;
      let dy = other.posVel.y - pred.posVel.y;
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

    let wallMargin = max(20.0, min(worldW, worldH) * 0.09);
    var wallAvoid = vec2f(0.0, 0.0);
    if (pred.posVel.x < wallMargin) {
      wallAvoid.x = wallAvoid.x + (1.0 - pred.posVel.x / wallMargin);
    } else if (pred.posVel.x > worldW - wallMargin) {
      wallAvoid.x = wallAvoid.x - (1.0 - (worldW - pred.posVel.x) / wallMargin);
    }
    if (pred.posVel.y < wallMargin) {
      wallAvoid.y = wallAvoid.y + (1.0 - pred.posVel.y / wallMargin);
    } else if (pred.posVel.y > worldH - wallMargin) {
      wallAvoid.y = wallAvoid.y - (1.0 - (worldH - pred.posVel.y) / wallMargin);
    }

    var desiredDir = chaseDir + separate * predatorSeparationWeight + wallAvoid * 0.95;
    var desiredHeading = pred.headingTimers.x;
    if (length(desiredDir) > 0.0001) {
      desiredHeading = atan2(desiredDir.y, desiredDir.x);
    }

    let strikeWindow = clamp(1.0 - targetDistance / max(params.boidA.z * select(1.3, 1.5, isApex), 1.0), 0.0, 1.0);
    let canStrike = targetIndex >= 0 && pred.headingTimers.w <= 0.0 && pred.headingTimers.z > 0.22;
    if (canStrike && strikeWindow > 0.36) {
      pred.aux.x = (0.28 + 0.24 * predatorSprintBoost) * select(1.0, 1.35, isApex);
      pred.headingTimers.w = 0.9 + 0.55 * (1.0 - pred.headingTimers.z) * select(1.0, 1.2, isApex);
    }
    let sprintActive = pred.aux.x > 0.0;
    let staminaDelta = select(
      dt * 0.26 * select(1.0, 0.8, isApex),
      -dt * (0.48 + 0.22 * predatorSprintBoost) * select(1.0, 1.15, isApex),
      sprintActive,
    );
    pred.headingTimers.z = clamp(pred.headingTimers.z + staminaDelta, 0.0, 1.0);
    let fatigueFactor = mix(select(0.64, 0.55, isApex), 1.0, pred.headingTimers.z);
    let sprintFactor = select(
      1.0,
      1.0 + predatorSprintBoost * (0.4 + 0.4 * pred.headingTimers.z) * select(1.0, 1.45, isApex),
      sprintActive,
    );
    let apexCruiseScale = select(1.0, 0.86, isApex);
    let predatorSpeed = predatorCruiseSpeed * apexCruiseScale * fatigueFactor * sprintFactor;

    let headingDelta = shortest_angle_delta(pred.headingTimers.x, desiredHeading);
    let desiredTurnRate = headingDelta / max(dt, 0.0001);
    let turnRateError = desiredTurnRate - pred.aux.w;
    let turnBoost = select(1.0, 1.0 + 0.28 * predatorSprintBoost, sprintActive);
    let maxRateDelta = predatorMaxTurnAcceleration * turnBoost * dt;
    var nextTurnRate = pred.aux.w + clamp(turnRateError, -maxRateDelta, maxRateDelta);
    nextTurnRate = clamp(nextTurnRate, -predatorMaxTurnRate * turnBoost, predatorMaxTurnRate * turnBoost);
    let nextHeading = normalize_angle(pred.headingTimers.x + nextTurnRate * dt);

    pred.headingTimers.x = nextHeading;
    pred.aux.w = nextTurnRate;

    var nextVel = vec2f(cos(nextHeading), sin(nextHeading)) * predatorSpeed;
    var nextPos = pred.posVel.xy + nextVel * frameScale;
    var boundedHeading = nextHeading;
    var boundedTurnRate = nextTurnRate;
    if (nextPos.x <= 0.0 || nextPos.x >= worldW) {
      nextVel.x = -nextVel.x;
      boundedHeading = atan2(nextVel.y, nextVel.x);
      boundedTurnRate = 0.0;
    }
    if (nextPos.y <= 0.0 || nextPos.y >= worldH) {
      nextVel.y = -nextVel.y;
      boundedHeading = atan2(nextVel.y, nextVel.x);
      boundedTurnRate = 0.0;
    }
    nextPos = vec2f(
      clamp_coordinate(nextPos.x, worldW),
      clamp_coordinate(nextPos.y, worldH),
    );
    pred.posVel.x = nextPos.x;
    pred.posVel.y = nextPos.y;
    pred.posVel.z = nextVel.x;
    pred.posVel.w = nextVel.y;
    pred.headingTimers.x = boundedHeading;
    pred.aux.w = boundedTurnRate;
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

const occupancyClearShader = `
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
  lifeB: vec4f,
};

@group(0) @binding(0) var<storage, read_write> occupancyClaims: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(${HEATMAP_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let sampleCount = u32(params.counts.w);
  if (gid.x >= sampleCount || gid.x >= ${HEATMAP_MAX_POINTS}u) {
    return;
  }
  atomicStore(&occupancyClaims[gid.x], 0u);
}
`;

const gridBuildShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
  bio: vec4f,
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
  lifeB: vec4f,
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
struct PheromoneSample {
  values: vec4f,
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
  lifeB: vec4f,
};

@group(0) @binding(0) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(2) var<storage, read> pheromonePrev: array<PheromoneSample>;
@group(0) @binding(3) var<storage, read_write> pheromoneNext: array<PheromoneSample>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<storage, read_write> occupancyClaims: array<atomic<u32>>;

const PHEROMONE_MAX_POINTS: u32 = ${HEATMAP_MAX_POINTS}u;
const TAU: f32 = 6.28318530718;

fn bounded_abs_delta(a: f32, b: f32) -> f32 {
  return abs(a - b);
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

fn read_pheromone(x: i32, y: i32, cols: u32, rows: u32, sampleCount: u32) -> vec4f {
  if (sampleCount == 0u || cols == 0u || rows == 0u) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  if (x < 0 || y < 0 || x >= i32(cols) || y >= i32(rows)) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  let idx = u32(y) * cols + u32(x);
  if (idx >= sampleCount || idx >= PHEROMONE_MAX_POINTS) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  return pheromonePrev[idx].values;
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
  let spacing = max(params.heatmapA.x, 0.75);
  let cols = max(1u, u32(floor(worldW / spacing)));
  let rows = max(1u, u32(floor(worldH / spacing)));
  let sx = idx % cols;
  let sy = idx / cols;
  let samplePos = vec2f((f32(sx) + 0.5) * spacing, (f32(sy) + 0.5) * spacing);
  let center = read_pheromone(i32(sx), i32(sy), cols, rows, sampleCount);

  let diffuseMix = clamp(params.boidB.z, 0.0, 1.0);
  let fearDiffuseMix = clamp(params.lifeB.z, 0.0, 1.0);
  let decay = clamp(params.boidB.w, 0.0, 0.95);
  let depositAmount = max(params.lifeA.x, 0.0);
  let nodeSourceStrength = max(params.lifeA.y, 0.0);
  let predatorCount = u32(params.counts.y);
  let fearStrength = max(params.lifeB.x, 0.0);
  let fearRadius = max(params.lifeB.y, spacing * 2.0);
  let fearRadiusSq = fearRadius * fearRadius;

  var neighborSum = vec4f(0.0, 0.0, 0.0, 0.0);
  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      neighborSum = neighborSum + read_pheromone(i32(sx) + ox, i32(sy) + oy, cols, rows, sampleCount);
    }
  }
  let neighborMean = neighborSum * (1.0 / 9.0);
  var trail = center.x + (neighborMean.x - center.x) * diffuseMix;
  var fear = center.y + (neighborMean.y - center.y) * fearDiffuseMix;
  var food = center.z + (neighborMean.z - center.z) * diffuseMix;
  var hazard = center.w + (neighborMean.w - center.w) * diffuseMix;

  let claim = atomicLoad(&occupancyClaims[idx]);
  if (claim > 0u) {
    let claimBoost = min(12.0, log2(1.0 + f32(claim)));
    trail = trail + depositAmount * claimBoost * 0.09;
  }

  if (nodeSourceStrength > 0.00001) {
    let t = f32(params.counts.z) * 0.002;
    for (var k: u32 = 0u; k < 6u; k = k + 1u) {
      let fk = f32(k);
      let orbit = t * (0.52 + fk * 0.06) + fk * (TAU / 6.0);
      let node = vec2f(
        worldW * (0.5 + 0.34 * cos(orbit + 0.45 * sin(t * 0.33 + fk))),
        worldH * (0.5 + 0.28 * sin(orbit * 0.92 + fk * 0.37)),
      );
      let dx = bounded_abs_delta(samplePos.x, node.x);
      let dy = bounded_abs_delta(samplePos.y, node.y);
      let distSq = dx * dx + dy * dy;
      let spread = 1350.0 + 320.0 * fk;
      trail = trail + exp(-distSq / spread) * nodeSourceStrength * 0.34;
    }
  }

  if (fearStrength > 0.00001 && predatorCount > 0u) {
    let fearSpread = max(fearRadiusSq * 0.7, 1.0);
    let hazardSpread = max(fearRadiusSq * 0.42, 1.0);
    for (var p: u32 = 0u; p < predatorCount; p = p + 1u) {
      let pred = predators[p];
      let dx = bounded_abs_delta(samplePos.x, pred.posVel.x);
      let dy = bounded_abs_delta(samplePos.y, pred.posVel.y);
      let distSq = dx * dx + dy * dy;
      if (distSq > fearRadiusSq * 4.0) {
        continue;
      }
      let apexScale = select(1.0, 1.35, pred.aux.y > 0.5);
      fear = fear + exp(-distSq / fearSpread) * fearStrength * 0.72 * apexScale;
      hazard = hazard + exp(-distSq / hazardSpread) * fearStrength * 0.08 * apexScale;
    }
  }

  trail = clamp(trail * (1.0 - decay), 0.0, 9.0);
  fear = clamp(fear * (1.0 - decay * 0.92), 0.0, 9.0);
  food = clamp(food * (1.0 - decay), 0.0, 9.0);
  hazard = clamp(hazard * (1.0 - decay), 0.0, 9.0);
  pheromoneNext[idx].values = vec4f(trail, fear, food, hazard);
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
  lifeB: vec4f,
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
  lifeB: vec4f,
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
    pred.headingTimers.z = clamp(pred.headingTimers.z + 0.1, 0.0, 1.0);
    pred.headingTimers.w = max(pred.headingTimers.w, 0.65);
    pred.aux.z = -1.0;
    pred.aux.x = 0.0;
    pred.headingTimers.y = 0.0;

    predators[p] = pred;
  }
}
`;

const boidRenderShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
  bio: vec4f,
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
  lifeB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<uniform> params: SimParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VSOut {
  let boid = boids[instanceIndex];
  let _unusedVertex = vertexIndex;
  let world = boid.posVel.xy;

  let clipX = world.x / params.world.x * 2.0 - 1.0;
  let clipY = 1.0 - world.y / params.world.y * 2.0;

  let speedRatio = clamp(length(boid.posVel.zw) / max(params.boidA.w, 0.0001), 0.0, 1.0);
  let moved = boid.headingTurn.y > 0.5;
  let baseColor = select(vec3f(0.26, 0.42, 0.66), vec3f(0.55, 0.98, 0.82), moved);

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.color = mix(baseColor, vec3f(0.96, 0.98, 1.0), speedRatio * 0.25);
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
  lifeB: vec4f,
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

const pheromoneRenderShader = `
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
  lifeB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec3f,
  @location(2) alpha: f32,
};

@group(0) @binding(0) var<storage, read> pheromones: array<PheromoneSample>;
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

  let sample = pheromones[instanceIndex].values;
  let trail = max(0.0, sample.x);
  let fear = max(0.0, sample.y);
  let food = max(0.0, sample.z);
  let hazard = max(0.0, sample.w);

  let total = trail + fear + food + hazard;
  let intensity = 1.0 - exp(-total * 0.55);
  let trailTone = clamp(log2(1.0 + trail * 18.0) * 0.24, 0.0, 1.0);
  let fearTone = clamp(fear * 0.44, 0.0, 1.0);
  let foodTone = clamp(food * 0.52, 0.0, 1.0);
  let hazardTone = clamp(hazard * 0.48, 0.0, 1.0);

  let base = vec3f(0.02, 0.05, 0.14);
  var color = base;
  color = color + vec3f(0.12, 0.90, 0.98) * trailTone;
  color = color + vec3f(1.00, 0.24, 0.32) * fearTone;
  color = color + vec3f(0.96, 0.88, 0.20) * foodTone;
  color = color + vec3f(1.00, 0.56, 0.12) * hazardTone;
  let blend = 0.28 + 0.72 * smoothstep(0.0, 0.9, intensity);
  color = mix(base, min(color, vec3f(1.0, 1.0, 1.0)), blend);

  let clipX = world.x / worldW * 2.0 - 1.0;
  let clipY = 1.0 - world.y / worldH * 2.0;

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.local = local;
  out.color = color;
  out.alpha = 0.22 + 0.78 * smoothstep(0.0, 1.0, intensity);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let distSq = dot(in.local, in.local);
  if (distSq > 1.0) {
    discard;
  }
  let edge = 1.0 - smoothstep(0.62, 1.0, distSq);
  let alpha = in.alpha * (0.33 + 0.67 * edge);
  return vec4f(in.color * alpha, alpha);
}
`;

const heatmapSampleComputeShader = `
struct BoidState {
  posVel: vec4f,
  headingTurn: vec4f,
  bio: vec4f,
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
  lifeB: vec4f,
};

@group(0) @binding(0) var<storage, read> boids: array<BoidState>;
@group(0) @binding(1) var<storage, read> heatPrev: array<HeatSample>;
@group(0) @binding(2) var<storage, read_write> heatNext: array<HeatSample>;
@group(0) @binding(3) var<uniform> params: SimParams;

const MIN_SAMPLES: u32 = 96u;
const MAX_SAMPLES: u32 = 768u;

fn bounded_abs_delta(a: f32, b: f32) -> f32 {
  return abs(a - b);
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
    let dx = bounded_abs_delta(samplePos.x, boid.posVel.x);
    let dy = bounded_abs_delta(samplePos.y, boid.posVel.y);
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
  lifeB: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec3f,
};

@group(0) @binding(0) var<storage, read> predators: array<PredatorState>;
@group(0) @binding(1) var<uniform> params: SimParams;

fn predator_vertex(vertexIndex: u32) -> vec2f {
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
  let predator = predators[instanceIndex];
  let local = predator_vertex(vertexIndex);
  let world = predator.posVel.xy + local * 2.15;

  let clipX = world.x / params.world.x * 2.0 - 1.0;
  let clipY = 1.0 - world.y / params.world.y * 2.0;

  var out: VSOut;
  out.position = vec4f(clipX, clipY, 0.0, 1.0);
  out.local = local;
  out.color = vec3f(1.0, 0.2, 0.25);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  if (dot(in.local, in.local) > 1.0) {
    discard;
  }
  return vec4f(in.color, 1.0);
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
  const spacing = getHeatmapDotSpacingWorld();
  const cols = Math.max(1, Math.floor(worldWidth / spacing));
  const rows = Math.max(1, Math.floor(worldHeight / spacing));
  for (let i = 0; i < BOID_COUNT; i += 1) {
    const heading = rand(0, Math.PI * 2);
    const speed = Math.max(0.2, Math.min(1.6, config.minSpeed * 1.25));
    const base = i * BOID_FLOATS;
    const cellX = Math.floor(rand(0, cols));
    const cellY = Math.floor(rand(0, rows));
    const px = (cellX + 0.5) * spacing;
    const py = (cellY + 0.5) * spacing;

    data[base + 0] = px;
    data[base + 1] = py;
    data[base + 2] = Math.cos(heading) * speed;
    data[base + 3] = Math.sin(heading) * speed;
    data[base + 4] = heading;
    data[base + 5] = 0;
    data[base + 6] = 1;
    data[base + 7] = 1;
    data[base + 8] = 0;
    data[base + 9] = 0;
    data[base + 10] = 0;
    data[base + 11] = 0;
  }
  return data;
}

function createInitialPredatorData() {
  const data = new Float32Array(PREDATOR_COUNT * PREDATOR_FLOATS);
  const predatorSpeed = config.maxSpeed * config.predatorSpeedFactor;

  for (let i = 0; i < PREDATOR_COUNT; i += 1) {
    const heading = rand(0, Math.PI * 2);
    const base = i * PREDATOR_FLOATS;
    const predatorType = 0;
    const speedScale = 1;

    data[base + 0] = rand(0, worldWidth);
    data[base + 1] = rand(0, worldHeight);
    data[base + 2] = Math.cos(heading) * predatorSpeed * speedScale;
    data[base + 3] = Math.sin(heading) * predatorSpeed * speedScale;

    data[base + 4] = heading;
    data[base + 5] = 0;
    data[base + 6] = 1;
    data[base + 7] = rand(0, 0.3);

    data[base + 8] = 0;
    data[base + 9] = predatorType;
    data[base + 10] = -1;
    data[base + 11] = 0;
  }

  return data;
}

function updateTextState(mode = 'running') {
  lastTextState = JSON.stringify({
    mode,
    renderer: 'webgpu',
    boidNeighborSearch: 'grid_local_neighbors',
    lifeFeatures: [
      'peer_cohesion',
      'peer_separation',
      'turn_rate_limited_steering',
      'trail_gradient_following',
      'trail_diffusion_decay',
    ],
    dynamicSystems: [
      'flocking_flow_fields',
      'bounded_edges',
      'crowding_feedback',
    ],
    boundaryMode: 'solid_bounce',
    coordinateSystem: 'origin top-left, +x right, +y down',
    viewport: { width: worldWidth, height: worldHeight },
    particleCount: BOID_COUNT,
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
    trailDeposit: config.physDeposit,
    sensorOffset: config.physSensorOffset,
    trailDiffuseMix: config.physDiffuseMix,
    trailDecay: config.physDecay,
    turnRateLimitDegPerSec: (config.maxTurnRate * 180) / Math.PI,
    physSensorAngleDeg: (config.physSensorAngle * 180) / Math.PI,
    physSensorOffset: config.physSensorOffset,
    physSensorWidth: config.physSensorWidth,
    physRotateAngleDeg: (config.maxTurnRate * 180) / Math.PI,
    physStepSize: config.physStepSize,
    boidCruiseSpeed: Math.max(0.2, Math.min(1.6, config.minSpeed * 1.25)),
    physDeposit: config.physDeposit,
    physDiffuseMix: config.physDiffuseMix,
    physDecay: config.physDecay,
    physMode: config.physModeRepel > 0.5 ? 'repel' : 'attract',
    nodeSourceStrength: config.physNodeSourceStrength,
    gridCellSize: GRID_CELL_SIZE,
    gridMaxCells: GRID_MAX_CELLS,
    gridCellCapacity: GRID_CELL_CAPACITY,
    maxTurnAccelerationDegPerSec2: Number(turnAccelRange.value),
    minSpeed: Number(minSpeedRange.value),
    avgSpeed: ecologyStats.avgSpeed,
    avgPredatorSpeed: ecologyStats.avgPredatorSpeed,
    boidsNearPredatorsFraction: ecologyStats.boidsNearPredatorsFraction,
    predatorSamples: ecologyStats.predatorSamples,
    movedFraction: ecologyStats.movedFraction,
    movesPerSecond: ecologyStats.movesPerSecond,
    blockedMoveRate: ecologyStats.blockedMoveRate,
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
  const boidCruiseSpeed = Math.max(0.2, Math.min(1.6, config.minSpeed * 1.25));
  return new Float32Array([
    worldWidth, worldHeight, dtSeconds, frameScale,
    BOID_COUNT, PREDATOR_COUNT, frameIndex, heatmapPointCount,
    config.physSensorAngle, config.physSensorOffset, config.physSensorWidth, boidCruiseSpeed,
    config.maxTurnRate, config.alignWeight, config.physDiffuseMix, config.physDecay,
    config.maxTurnAcceleration, config.cohesionWeight, config.separationWeight, 0.72,
    config.predatorAvoidWeight, config.predatorAttentionSeconds, config.perceptionRadius, config.separationRadius,
    config.predatorSprintBoost, config.predatorSpeedFactor, predatorMaxTurnRate, predatorMaxTurnAcceleration,
    getHeatmapDotSpacingWorld(), getHeatmapDotRadiusWorld(), getHeatmapDiffuseRadiusWorld(), config.heatmapSampleBudget,
    config.heatmapTrendGain, config.heatmapTrendDeadband, config.pherDiffusion, config.pherDecay,
    config.physDeposit, config.physNodeSourceStrength, config.predatorAvoidRadius, 0.25 + config.predatorFearStrength * 0.9,
    config.predatorFearStrength, config.predatorFearRadius, config.physDiffuseMix, config.mutationRate,
  ]);
}

function writeParams(dtSeconds) {
  const params = createParamsArray(dtSeconds);
  gpu.device.queue.writeBuffer(gpu.paramsBuffer, 0, params);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function maybeScheduleEcologyReadback(timestampMs) {
  if (!gpu || ecologyStats.readbackPending) {
    return;
  }
  if (timestampMs - ecologyStats.lastReadbackAtMs < ECOLOGY_READBACK_INTERVAL_MS) {
    return;
  }

  ecologyStats.readbackPending = true;
  ecologyStats.lastReadbackAtMs = timestampMs;
  const sampleBoids = Math.min(BOID_STATS_SAMPLE_COUNT, BOID_COUNT);
  const boidReadbackBytes = sampleBoids * BOID_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const predatorReadbackBytes = PREDATOR_COUNT * PREDATOR_FLOATS * Float32Array.BYTES_PER_ELEMENT;
  const counterReadbackBytes = SIM_COUNTER_COUNT * Uint32Array.BYTES_PER_ELEMENT;
  const readbackEncoder = gpu.device.createCommandEncoder();
  readbackEncoder.copyBufferToBuffer(
    gpu.boidBuffers[currentBoidBufferIndex],
    0,
    gpu.ecologyReadbackBuffer,
    0,
    boidReadbackBytes,
  );
  readbackEncoder.copyBufferToBuffer(
    gpu.predatorBuffer,
    0,
    gpu.predatorReadbackBuffer,
    0,
    predatorReadbackBytes,
  );
  readbackEncoder.copyBufferToBuffer(
    gpu.simCountersBuffer,
    0,
    gpu.counterReadbackBuffer,
    0,
    counterReadbackBytes,
  );
  gpu.device.queue.submit([readbackEncoder.finish()]);

  Promise.all([
    gpu.ecologyReadbackBuffer.mapAsync(GPUMapMode.READ),
    gpu.predatorReadbackBuffer.mapAsync(GPUMapMode.READ),
    gpu.counterReadbackBuffer.mapAsync(GPUMapMode.READ),
  ])
    .then(() => {
      const boidSample = new Float32Array(gpu.ecologyReadbackBuffer.getMappedRange().slice(0));
      const predatorSample = new Float32Array(gpu.predatorReadbackBuffer.getMappedRange().slice(0));
      const counters = new Uint32Array(gpu.counterReadbackBuffer.getMappedRange().slice(0));
      gpu.ecologyReadbackBuffer.unmap();
      gpu.predatorReadbackBuffer.unmap();
      gpu.counterReadbackBuffer.unmap();

      const count = Math.max(1, Math.min(sampleBoids, Math.floor(boidSample.length / BOID_FLOATS)));
      let speedTotal = 0;
      let movedCount = 0;
      for (let i = 0; i < count; i += 1) {
        const base = i * BOID_FLOATS;
        const vx = boidSample[base + 2] || 0;
        const vy = boidSample[base + 3] || 0;
        speedTotal += Math.hypot(vx, vy);
        if ((boidSample[base + 5] || 0) > 0.5) {
          movedCount += 1;
        }
      }

      const predatorCount = Math.min(PREDATOR_COUNT, Math.floor(predatorSample.length / PREDATOR_FLOATS));
      let predatorSpeedTotal = 0;
      const predatorSamples = [];
      for (let i = 0; i < predatorCount; i += 1) {
        const base = i * PREDATOR_FLOATS;
        const px = predatorSample[base + 0] || 0;
        const py = predatorSample[base + 1] || 0;
        const vx = predatorSample[base + 2] || 0;
        const vy = predatorSample[base + 3] || 0;
        const speed = Math.hypot(vx, vy);
        predatorSpeedTotal += speed;
        predatorSamples.push({
          id: i,
          x: Number(px.toFixed(2)),
          y: Number(py.toFixed(2)),
          speed: Number(speed.toFixed(3)),
        });
      }

      let nearPredatorCount = 0;
      const avoidRadius = Math.max(8, config.predatorAvoidRadius);
      const avoidRadiusSq = avoidRadius * avoidRadius;
      for (let i = 0; i < count; i += 1) {
        const base = i * BOID_FLOATS;
        const bx = boidSample[base + 0] || 0;
        const by = boidSample[base + 1] || 0;
        let nearAny = false;
        for (let p = 0; p < predatorCount; p += 1) {
          const pBase = p * PREDATOR_FLOATS;
          const px = predatorSample[pBase + 0] || 0;
          const py = predatorSample[pBase + 1] || 0;
          let dx = Math.abs(px - bx);
          let dy = Math.abs(py - by);
          if (dx > worldWidth * 0.5) dx = worldWidth - dx;
          if (dy > worldHeight * 0.5) dy = worldHeight - dy;
          if (dx * dx + dy * dy <= avoidRadiusSq) {
            nearAny = true;
            break;
          }
        }
        if (nearAny) {
          nearPredatorCount += 1;
        }
      }

      const nowMs = performance.now();
      if (ecologyStats.lastCounterSampleAtMs === 0) {
        ecologyStats.lastCounterSampleAtMs = nowMs - ECOLOGY_READBACK_INTERVAL_MS;
      }
      const elapsedSeconds = Math.max(0.001, (nowMs - ecologyStats.lastCounterSampleAtMs) / 1000);
      ecologyStats.lastCounterSampleAtMs = nowMs;
      const movedCounter = counters[0] || 0;
      const blockedCounter = counters[1] || 0;

      const alpha = 0.22;
      ecologyStats.avgSpeed = ecologyStats.avgSpeed + (speedTotal / count - ecologyStats.avgSpeed) * alpha;
      const predatorAvg = predatorCount > 0 ? predatorSpeedTotal / predatorCount : 0;
      ecologyStats.avgPredatorSpeed =
        ecologyStats.avgPredatorSpeed + (predatorAvg - ecologyStats.avgPredatorSpeed) * alpha;
      ecologyStats.predatorSamples = predatorSamples;
      ecologyStats.boidsNearPredatorsFraction =
        ecologyStats.boidsNearPredatorsFraction +
        (nearPredatorCount / count - ecologyStats.boidsNearPredatorsFraction) * alpha;
      ecologyStats.movedFraction = ecologyStats.movedFraction + (movedCount / count - ecologyStats.movedFraction) * alpha;
      ecologyStats.movesPerSecond =
        ecologyStats.movesPerSecond + (movedCounter / elapsedSeconds - ecologyStats.movesPerSecond) * alpha;
      ecologyStats.blockedMoveRate =
        ecologyStats.blockedMoveRate +
        (blockedCounter / Math.max(1, movedCounter + blockedCounter) - ecologyStats.blockedMoveRate) * alpha;

      gpu.device.queue.writeBuffer(gpu.simCountersBuffer, 0, new Uint32Array(SIM_COUNTER_COUNT));
      ecologyStats.readbackPending = false;
    })
    .catch(() => {
      if (gpu?.ecologyReadbackBuffer?.mapState === 'mapped') {
        gpu.ecologyReadbackBuffer.unmap();
      }
      if (gpu?.predatorReadbackBuffer?.mapState === 'mapped') {
        gpu.predatorReadbackBuffer.unmap();
      }
      if (gpu?.counterReadbackBuffer?.mapState === 'mapped') {
        gpu.counterReadbackBuffer.unmap();
      }
      ecologyStats.readbackPending = false;
    });
}

function reseedSimulation() {
  if (!gpu) {
    return;
  }

  const boidData = createInitialBoidData();
  const predatorData = createInitialPredatorData();
  const zeroFlags = new Uint32Array(PREDATOR_COUNT);
  const zeroCounters = new Uint32Array(SIM_COUNTER_COUNT);
  const zeroGridCounts = new Uint32Array(GRID_MAX_CELLS);
  const zeroOccupancy = new Uint32Array(HEATMAP_MAX_POINTS);
  const zeroHeat = new Float32Array(HEATMAP_MAX_POINTS * 4);
  const zeroPheromone = new Float32Array(HEATMAP_MAX_POINTS * PHEROMONE_FLOATS);

  gpu.device.queue.writeBuffer(gpu.boidBuffers[0], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.boidBuffers[1], 0, boidData);
  gpu.device.queue.writeBuffer(gpu.predatorBuffer, 0, predatorData);
  gpu.device.queue.writeBuffer(gpu.caughtFlagsBuffer, 0, zeroFlags);
  gpu.device.queue.writeBuffer(gpu.simCountersBuffer, 0, zeroCounters);
  gpu.device.queue.writeBuffer(gpu.gridCellCountBuffer, 0, zeroGridCounts);
  gpu.device.queue.writeBuffer(gpu.occupancyBuffer, 0, zeroOccupancy);
  gpu.device.queue.writeBuffer(gpu.heatmapBuffers[0], 0, zeroHeat);
  gpu.device.queue.writeBuffer(gpu.heatmapBuffers[1], 0, zeroHeat);
  gpu.device.queue.writeBuffer(gpu.pheromoneBuffers[0], 0, zeroPheromone);
  gpu.device.queue.writeBuffer(gpu.pheromoneBuffers[1], 0, zeroPheromone);

  frameIndex = 0;
  accumulator = 0;
  currentBoidBufferIndex = 0;
  currentHeatmapBufferIndex = 0;
  currentPheromoneBufferIndex = 0;
  ecologyStats.avgSpeed = Math.max(0.2, Math.min(1.6, config.minSpeed * 1.25));
  ecologyStats.avgPredatorSpeed = 0;
  ecologyStats.boidsNearPredatorsFraction = 0;
  ecologyStats.predatorSamples = [];
  ecologyStats.movedFraction = 0;
  ecologyStats.movesPerSecond = 0;
  ecologyStats.blockedMoveRate = 0;
  ecologyStats.lastCounterSampleAtMs = performance.now();
  ecologyStats.readbackPending = false;
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
  const gridBuildBindGroup = gpu.gridBuildBindGroups[currentBoidBufferIndex];
  if (!boidUpdateBindGroup || !gpu.occupancyClearBindGroup || !gpu.gridClearBindGroup || !gridBuildBindGroup) {
    throw new Error(`Missing compute bind group for boid buffer index ${currentBoidBufferIndex}.`);
  }

  writeParams(dtSeconds);

  const encoder = gpu.device.createCommandEncoder();
  const simPass = encoder.beginComputePass();

  try {
    simPass.setPipeline(gpu.occupancyClearPipeline);
    simPass.setBindGroup(0, gpu.occupancyClearBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(heatmapPointCount / HEATMAP_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.gridClearPipeline);
    simPass.setBindGroup(0, gpu.gridClearBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(GRID_MAX_CELLS / GRID_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.gridBuildPipeline);
    simPass.setBindGroup(0, gridBuildBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / BOID_WORKGROUP_SIZE));

    simPass.setPipeline(gpu.boidUpdatePipeline);
    simPass.setBindGroup(0, boidUpdateBindGroup);
    simPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / BOID_WORKGROUP_SIZE));
  } finally {
    simPass.end();
  }

  currentBoidBufferIndex = 1 - currentBoidBufferIndex;
  frameIndex += 1;

  const predatorUpdateBindGroup = gpu.predatorUpdateBindGroups[currentBoidBufferIndex];
  if (!predatorUpdateBindGroup) {
    throw new Error('Missing predator update bind group.');
  }
  const predatorPass = encoder.beginComputePass();
  try {
    predatorPass.setPipeline(gpu.predatorUpdatePipeline);
    predatorPass.setBindGroup(0, predatorUpdateBindGroup);
    predatorPass.dispatchWorkgroups(1);
  } finally {
    predatorPass.end();
  }

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
  heatmapCpuMs += performance.now() - heatmapStart;

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
  const pheromoneRenderBindGroup = gpu.pheromoneRenderBindGroups[currentPheromoneBufferIndex];
  const predatorRenderBindGroup = gpu.predatorRenderBindGroup;
  if (!boidRenderBindGroup || !heatmapRenderBindGroup || !pheromoneRenderBindGroup || !predatorRenderBindGroup) {
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
    } else if (renderMode === 'pheromone') {
      renderPass.setPipeline(gpu.pheromoneRenderPipeline);
      renderPass.setBindGroup(0, pheromoneRenderBindGroup);
      renderPass.draw(6, heatmapPointCount, 0, 0);
    } else {
      renderPass.setPipeline(gpu.boidRenderPipeline);
      renderPass.setBindGroup(0, boidRenderBindGroup);
      renderPass.draw(1, BOID_COUNT, 0, 0);
    }

    if (renderMode !== 'pheromone') {
      renderPass.setPipeline(gpu.predatorRenderPipeline);
      renderPass.setBindGroup(0, predatorRenderBindGroup);
      renderPass.draw(6, PREDATOR_COUNT, 0, 0);
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
    maybeScheduleEcologyReadback(timestamp);

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

  maybeScheduleEcologyReadback(performance.now());
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
  if (renderMode === 'heatmap') {
    viewToggleButton.textContent = 'View: Heatmap';
  } else if (renderMode === 'pheromone') {
    viewToggleButton.textContent = 'View: Pheromone';
  } else {
    viewToggleButton.textContent = 'View: Arrows';
  }
  viewToggleButton.setAttribute('aria-pressed', renderMode === 'boids' ? 'false' : 'true');
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
  const occupancyBufferBytes = HEATMAP_MAX_POINTS * Uint32Array.BYTES_PER_ELEMENT;
  const gridCellCountBytes = GRID_MAX_CELLS * Uint32Array.BYTES_PER_ELEMENT;
  const gridBoidIndexBytes = GRID_MAX_CELLS * GRID_CELL_CAPACITY * Uint32Array.BYTES_PER_ELEMENT;
  const paramsBufferBytes = 44 * Float32Array.BYTES_PER_ELEMENT;
  const caughtFlagsBytes = PREDATOR_COUNT * Uint32Array.BYTES_PER_ELEMENT;
  const simCountersBytes = SIM_COUNTER_COUNT * Uint32Array.BYTES_PER_ELEMENT;
  const boidReadbackBytes = Math.min(BOID_STATS_SAMPLE_COUNT, BOID_COUNT) * BOID_FLOATS * Float32Array.BYTES_PER_ELEMENT;

  const boidBuffers = [
    device.createBuffer({
      size: boidBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    device.createBuffer({
      size: boidBufferBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
  ];

  const predatorBuffer = device.createBuffer({
    size: predatorBufferBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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

  const occupancyBuffer = device.createBuffer({
    size: occupancyBufferBytes,
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

  const simCountersBuffer = device.createBuffer({
    size: simCountersBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const paramsBuffer = device.createBuffer({
    size: paramsBufferBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const ecologyReadbackBuffer = device.createBuffer({
    size: boidReadbackBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const predatorReadbackBuffer = device.createBuffer({
    size: predatorBufferBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const counterReadbackBuffer = device.createBuffer({
    size: simCountersBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
  const occupancyClearModule = await createCheckedShaderModule('occupancyClearShader', occupancyClearShader);
  const gridBuildModule = await createCheckedShaderModule('gridBuildShader', gridBuildShader);
  const pheromoneUpdateModule = await createCheckedShaderModule('pheromoneUpdateShader', pheromoneUpdateShader);
  const clearCaughtModule = await createCheckedShaderModule('catchClearShader', catchClearShader);
  const predatorResolveModule = await createCheckedShaderModule('predatorResolveShader', predatorResolveShader);
  const boidRenderModule = await createCheckedShaderModule('boidRenderShader', boidRenderShader);
  const heatmapSampleComputeModule = await createCheckedShaderModule('heatmapSampleComputeShader', heatmapSampleComputeShader);
  const boidHeatmapRenderModule = await createCheckedShaderModule('boidHeatmapRenderShader', boidHeatmapRenderShader);
  const pheromoneRenderModule = await createCheckedShaderModule('pheromoneRenderShader', pheromoneRenderShader);
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

  const occupancyClearPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: occupancyClearModule,
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
      topology: 'point-list',
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

  const pheromoneRenderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: pheromoneRenderModule,
      entryPoint: 'vsMain',
    },
    fragment: {
      module: pheromoneRenderModule,
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
          { binding: 4, resource: { buffer: occupancyBuffer } },
          { binding: 5, resource: { buffer: gridCellCountBuffer } },
          { binding: 6, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
          { binding: 8, resource: { buffer: simCountersBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[0] } },
          { binding: 1, resource: { buffer: boidBuffers[1] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: occupancyBuffer } },
          { binding: 5, resource: { buffer: gridCellCountBuffer } },
          { binding: 6, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
          { binding: 8, resource: { buffer: simCountersBuffer } },
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
          { binding: 4, resource: { buffer: occupancyBuffer } },
          { binding: 5, resource: { buffer: gridCellCountBuffer } },
          { binding: 6, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
          { binding: 8, resource: { buffer: simCountersBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: boidUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: boidBuffers[1] } },
          { binding: 1, resource: { buffer: boidBuffers[0] } },
          { binding: 2, resource: { buffer: predatorBuffer } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: occupancyBuffer } },
          { binding: 5, resource: { buffer: gridCellCountBuffer } },
          { binding: 6, resource: { buffer: gridBoidIndexBuffer } },
          { binding: 7, resource: { buffer: paramsBuffer } },
          { binding: 8, resource: { buffer: simCountersBuffer } },
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

  const occupancyClearBindGroup = device.createBindGroup({
    layout: occupancyClearPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: occupancyBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
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
          { binding: 0, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
          { binding: 5, resource: { buffer: occupancyBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
          { binding: 5, resource: { buffer: occupancyBuffer } },
        ],
      }),
    ],
    [
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
          { binding: 5, resource: { buffer: occupancyBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: pheromoneUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: predatorBuffer } },
          { binding: 2, resource: { buffer: pheromoneBuffers[1] } },
          { binding: 3, resource: { buffer: pheromoneBuffers[0] } },
          { binding: 4, resource: { buffer: paramsBuffer } },
          { binding: 5, resource: { buffer: occupancyBuffer } },
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

  const pheromoneRenderBindGroups = [
    device.createBindGroup({
      layout: pheromoneRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pheromoneBuffers[0] } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
    device.createBindGroup({
      layout: pheromoneRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pheromoneBuffers[1] } },
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
    occupancyBuffer,
    gridCellCountBuffer,
    gridBoidIndexBuffer,
    caughtFlagsBuffer,
    simCountersBuffer,
    paramsBuffer,
    ecologyReadbackBuffer,
    predatorReadbackBuffer,
    counterReadbackBuffer,
    boidUpdatePipeline,
    predatorUpdatePipeline,
    gridClearPipeline,
    occupancyClearPipeline,
    gridBuildPipeline,
    pheromoneUpdatePipeline,
    clearCaughtPipeline,
    predatorResolvePipeline,
    heatmapComputePipeline,
    boidRenderPipeline,
    boidHeatmapRenderPipeline,
    pheromoneRenderPipeline,
    predatorRenderPipeline,
    boidUpdateBindGroups,
    predatorUpdateBindGroups,
    gridClearBindGroup,
    occupancyClearBindGroup,
    gridBuildBindGroups,
    pheromoneUpdateBindGroups,
    heatmapComputeBindGroups,
    clearCaughtBindGroup,
    predatorResolveBindGroup,
    boidRenderBindGroups,
    boidHeatmapRenderBindGroups,
    pheromoneRenderBindGroups,
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
  pherTrailWeightValue.textContent = config.physDeposit.toFixed(2);
  pherFearWeightValue.textContent = config.physSensorOffset.toFixed(1);
  pherDiffusionValue.textContent = config.physDiffuseMix.toFixed(2);
  pherDecayValue.textContent = config.physDecay.toFixed(3);
  panicBoostValue.textContent = ((config.maxTurnRate * 180) / Math.PI).toFixed(0);
}

function updateEcologyControlLabels() {
  metabolismValue.textContent = config.metabolismIntensity.toFixed(2);
  boidSprintBoostValue.textContent = config.boidSprintBoost.toFixed(2);
  predatorSprintBoostValue.textContent = config.predatorSprintBoost.toFixed(2);
  confusionStrengthValue.textContent = config.confusionStrength.toFixed(2);
  mutationRateValue.textContent = config.mutationRate.toFixed(3);
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
  predatorAttentionValue.textContent = Number(predatorAttentionRange.value).toFixed(2);
  config.physNodeSourceStrength = Number(predatorAttentionRange.value);
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
  config.physDeposit = Number(pherTrailWeightRange.value);
  updateLifeFieldControlLabels();
});

pherFearWeightRange.addEventListener('input', () => {
  config.physSensorOffset = Number(pherFearWeightRange.value);
  updateLifeFieldControlLabels();
});

pherDiffusionRange.addEventListener('input', () => {
  config.physDiffuseMix = Number(pherDiffusionRange.value);
  updateLifeFieldControlLabels();
});

pherDecayRange.addEventListener('input', () => {
  config.physDecay = Number(pherDecayRange.value);
  updateLifeFieldControlLabels();
});

panicBoostRange.addEventListener('input', () => {
  config.maxTurnRate = (Number(panicBoostRange.value) * Math.PI) / 180;
  updateLifeFieldControlLabels();
});

metabolismRange.addEventListener('input', () => {
  config.metabolismIntensity = Number(metabolismRange.value);
  updateEcologyControlLabels();
});

boidSprintBoostRange.addEventListener('input', () => {
  config.boidSprintBoost = Number(boidSprintBoostRange.value);
  updateEcologyControlLabels();
});

predatorSprintBoostRange.addEventListener('input', () => {
  config.predatorSprintBoost = Number(predatorSprintBoostRange.value);
  updateEcologyControlLabels();
});

confusionStrengthRange.addEventListener('input', () => {
  config.confusionStrength = Number(confusionStrengthRange.value);
  updateEcologyControlLabels();
});

mutationRateRange.addEventListener('input', () => {
  config.mutationRate = Number(mutationRateRange.value);
  updateEcologyControlLabels();
});

viewToggleButton.addEventListener('click', () => {
  if (renderMode === 'boids') {
    renderMode = 'heatmap';
  } else if (renderMode === 'heatmap') {
    renderMode = 'pheromone';
  } else {
    renderMode = 'boids';
  }
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
predatorAttentionValue.textContent = Number(predatorAttentionRange.value).toFixed(2);
updateHeatmapControlLabels();
updateLifeFieldControlLabels();
updateEcologyControlLabels();
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
