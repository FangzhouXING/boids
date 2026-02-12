const canvas = document.getElementById('boidCanvas');
const ctx = canvas.getContext('2d');

const restartButton = document.getElementById('restartButton');
const turnAccelRange = document.getElementById('turnAccelRange');
const turnAccelValue = document.getElementById('turnAccelValue');
const minSpeedRange = document.getElementById('minSpeedRange');
const minSpeedValue = document.getElementById('minSpeedValue');
const predatorAttentionRange = document.getElementById('predatorAttentionRange');
const predatorAttentionValue = document.getElementById('predatorAttentionValue');

const dpr = Math.max(window.devicePixelRatio || 1, 1);
const FIXED_STEP = 1 / 60;
const BOID_COUNT = 200;
const PREDATOR_COUNT = 5;

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
  predatorSpeedFactor: 1.03,
  predatorPostCatchPauseMinSeconds: 3,
  predatorPostCatchPauseMaxSeconds: 10,
  predatorPauseSlowdownSeconds: 1.4,
  predatorAttentionSeconds: Number(predatorAttentionRange.value),
};

let boids = [];
let predators = [];
let worldWidth = 960;
let worldHeight = 560;
let lastFrameTime = 0;
let nextBoidId = 1;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function normalizeAngle(angle) {
  let result = angle;
  while (result <= -Math.PI) {
    result += Math.PI * 2;
  }
  while (result > Math.PI) {
    result -= Math.PI * 2;
  }
  return result;
}

function shortestAngleDelta(from, to) {
  return normalizeAngle(to - from);
}

function limitMagnitude(x, y, max) {
  const length = Math.hypot(x, y);
  if (length <= max || length === 0) {
    return { x, y };
  }
  const scale = max / length;
  return { x: x * scale, y: y * scale };
}

function wrapCoordinate(value, size) {
  if (value < 0) {
    return value + size;
  }
  if (value >= size) {
    return value - size;
  }
  return value;
}

function wrappedDelta(from, to, size) {
  let delta = to - from;
  const half = size * 0.5;
  if (delta > half) {
    delta -= size;
  } else if (delta < -half) {
    delta += size;
  }
  return delta;
}

function steerTowards(boid, targetX, targetY, forceCap = config.maxForce) {
  const magnitude = Math.hypot(targetX, targetY);
  if (magnitude < 0.0001) {
    return { x: 0, y: 0 };
  }

  const desiredX = (targetX / magnitude) * config.maxSpeed;
  const desiredY = (targetY / magnitude) * config.maxSpeed;
  const steerX = desiredX - boid.vx;
  const steerY = desiredY - boid.vy;
  return limitMagnitude(steerX, steerY, forceCap);
}

function makeBoid() {
  const startHeading = rand(0, Math.PI * 2);
  const startSpeed = rand(config.minSpeed, config.maxSpeed);
  return {
    id: nextBoidId++,
    x: rand(0, worldWidth),
    y: rand(0, worldHeight),
    vx: Math.cos(startHeading) * startSpeed,
    vy: Math.sin(startHeading) * startSpeed,
    heading: startHeading,
    turnRate: 0,
  };
}

function makePredator() {
  const heading = rand(0, Math.PI * 2);
  const speed = config.maxSpeed * 0.8;
  return {
    x: rand(0, worldWidth),
    y: rand(0, worldHeight),
    vx: Math.cos(heading) * speed,
    vy: Math.sin(heading) * speed,
    heading,
    targetId: null,
    attentionTimer: 0,
    pauseTimer: 0,
    pauseSlowdownTimer: 0,
    pauseStartVx: 0,
    pauseStartVy: 0,
  };
}

function initBoids(count) {
  boids = Array.from({ length: count }, makeBoid);
}

function initPredators(count) {
  predators = Array.from({ length: count }, makePredator);
}

function resizeCanvas() {
  const previousWidth = worldWidth;
  const previousHeight = worldHeight;

  const rect = canvas.getBoundingClientRect();
  worldWidth = Math.max(1, Math.floor(rect.width));
  worldHeight = Math.max(1, Math.floor(rect.height));

  canvas.width = Math.floor(worldWidth * dpr);
  canvas.height = Math.floor(worldHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (previousWidth === 0 || previousHeight === 0) {
    return;
  }

  const sx = worldWidth / previousWidth;
  const sy = worldHeight / previousHeight;

  for (const boid of boids) {
    boid.x = clamp(boid.x * sx, 0, worldWidth - 0.01);
    boid.y = clamp(boid.y * sy, 0, worldHeight - 0.01);
  }

  for (const predator of predators) {
    predator.x = clamp(predator.x * sx, 0, worldWidth - 0.01);
    predator.y = clamp(predator.y * sy, 0, worldHeight - 0.01);
  }
}

function updateBoids(dtSeconds) {
  const perceptionSq = config.perceptionRadius * config.perceptionRadius;
  const separationSq = config.separationRadius * config.separationRadius;
  const predatorAvoidSq = config.predatorAvoidRadius * config.predatorAvoidRadius;
  const next = new Array(boids.length);
  const frameScale = dtSeconds * 60;

  for (let i = 0; i < boids.length; i += 1) {
    const boid = boids[i];

    let alignX = 0;
    let alignY = 0;
    let cohesionX = 0;
    let cohesionY = 0;
    let separationX = 0;
    let separationY = 0;
    let neighbors = 0;

    for (let j = 0; j < boids.length; j += 1) {
      if (i === j) {
        continue;
      }

      const other = boids[j];
      const dx = wrappedDelta(boid.x, other.x, worldWidth);
      const dy = wrappedDelta(boid.y, other.y, worldHeight);
      const distSq = dx * dx + dy * dy;

      if (distSq <= 0 || distSq > perceptionSq) {
        continue;
      }

      neighbors += 1;
      alignX += other.vx;
      alignY += other.vy;
      cohesionX += dx;
      cohesionY += dy;

      if (distSq < separationSq) {
        const invDist = 1 / Math.sqrt(distSq);
        separationX -= dx * invDist;
        separationY -= dy * invDist;
      }
    }

    let predatorAvoidX = 0;
    let predatorAvoidY = 0;
    let predatorThreats = 0;

    for (const predator of predators) {
      const dx = wrappedDelta(boid.x, predator.x, worldWidth);
      const dy = wrappedDelta(boid.y, predator.y, worldHeight);
      const distSq = dx * dx + dy * dy;

      if (distSq <= 0 || distSq > predatorAvoidSq) {
        continue;
      }

      const invDist = 1 / Math.sqrt(distSq);
      predatorAvoidX -= dx * invDist;
      predatorAvoidY -= dy * invDist;
      predatorThreats += 1;
    }

    let desiredVx = boid.vx;
    let desiredVy = boid.vy;

    if (neighbors > 0) {
      alignX /= neighbors;
      alignY /= neighbors;
      cohesionX /= neighbors;
      cohesionY /= neighbors;
      separationX /= neighbors;
      separationY /= neighbors;

      const alignSteer = steerTowards(boid, alignX, alignY);
      const cohesionSteer = steerTowards(boid, cohesionX, cohesionY);
      const separationSteer = steerTowards(boid, separationX, separationY, config.maxForce * 1.6);

      desiredVx += alignSteer.x * config.alignWeight;
      desiredVy += alignSteer.y * config.alignWeight;
      desiredVx += cohesionSteer.x * config.cohesionWeight;
      desiredVy += cohesionSteer.y * config.cohesionWeight;
      desiredVx += separationSteer.x * config.separationWeight;
      desiredVy += separationSteer.y * config.separationWeight;
    }

    if (predatorThreats > 0) {
      predatorAvoidX /= predatorThreats;
      predatorAvoidY /= predatorThreats;
      const predatorSteer = steerTowards(boid, predatorAvoidX, predatorAvoidY, config.maxForce * 2.2);
      desiredVx += predatorSteer.x * config.predatorAvoidWeight;
      desiredVy += predatorSteer.y * config.predatorAvoidWeight;
    }

    const desiredLimited = limitMagnitude(desiredVx, desiredVy, config.maxSpeed);
    const desiredSpeed = Math.hypot(desiredLimited.x, desiredLimited.y);
    const speed = clamp(desiredSpeed, config.minSpeed, config.maxSpeed);

    const targetHeading = desiredSpeed > 0.0001 ? Math.atan2(desiredLimited.y, desiredLimited.x) : boid.heading;
    const headingDelta = shortestAngleDelta(boid.heading, targetHeading);
    const desiredTurnRate = headingDelta / Math.max(dtSeconds, 1e-4);
    const turnRateError = desiredTurnRate - boid.turnRate;
    const maxRateDelta = config.maxTurnAcceleration * dtSeconds;

    let nextTurnRate = boid.turnRate + clamp(turnRateError, -maxRateDelta, maxRateDelta);
    nextTurnRate = clamp(nextTurnRate, -config.maxTurnRate, config.maxTurnRate);
    const nextHeading = normalizeAngle(boid.heading + nextTurnRate * dtSeconds);

    const alignedVx = Math.cos(nextHeading) * speed;
    const alignedVy = Math.sin(nextHeading) * speed;

    const movedX = boid.x + alignedVx * frameScale;
    const movedY = boid.y + alignedVy * frameScale;

    next[i] = {
      id: boid.id,
      x: wrapCoordinate(movedX, worldWidth),
      y: wrapCoordinate(movedY, worldHeight),
      vx: alignedVx,
      vy: alignedVy,
      heading: nextHeading,
      turnRate: nextTurnRate,
    };
  }

  boids = next;
}

function findClosestBoidId(predator) {
  if (boids.length === 0) {
    return null;
  }

  let closestId = boids[0].id;
  let bestDistSq = Infinity;

  for (const boid of boids) {
    const dx = wrappedDelta(predator.x, boid.x, worldWidth);
    const dy = wrappedDelta(predator.y, boid.y, worldHeight);
    const distSq = dx * dx + dy * dy;
    if (distSq < bestDistSq) {
      bestDistSq = distSq;
      closestId = boid.id;
    }
  }

  return closestId;
}

function updatePredators(dtSeconds) {
  const frameScale = dtSeconds * 60;
  const predatorSpeed = config.maxSpeed * config.predatorSpeedFactor;
  const catchSq = config.predatorCatchRadius * config.predatorCatchRadius;
  const predatorSeparationSq = config.predatorSeparationRadius * config.predatorSeparationRadius;
  const caughtIds = new Set();

  for (const predator of predators) {
    if (predator.pauseTimer > 0) {
      const pauseStep = Math.min(dtSeconds, predator.pauseTimer);
      predator.pauseTimer = Math.max(0, predator.pauseTimer - pauseStep);

      let pauseVx = 0;
      let pauseVy = 0;
      if (predator.pauseSlowdownTimer > 0) {
        predator.pauseSlowdownTimer = Math.max(0, predator.pauseSlowdownTimer - pauseStep);
        const slowdownFactor = predator.pauseSlowdownTimer / Math.max(config.predatorPauseSlowdownSeconds, 1e-4);
        pauseVx = predator.pauseStartVx * slowdownFactor;
        pauseVy = predator.pauseStartVy * slowdownFactor;

        predator.x = wrapCoordinate(predator.x + pauseVx * (pauseStep * 60), worldWidth);
        predator.y = wrapCoordinate(predator.y + pauseVy * (pauseStep * 60), worldHeight);
      }

      predator.vx = pauseVx;
      predator.vy = pauseVy;
      continue;
    }

    predator.attentionTimer -= dtSeconds;

    if (predator.attentionTimer <= 0) {
      predator.targetId = findClosestBoidId(predator);
      predator.attentionTimer = config.predatorAttentionSeconds;
    }

    let target = null;
    if (predator.targetId !== null) {
      target = boids.find((boid) => boid.id === predator.targetId) || null;
    }

    let chaseX = Math.cos(predator.heading);
    let chaseY = Math.sin(predator.heading);
    if (target) {
      const dx = wrappedDelta(predator.x, target.x, worldWidth);
      const dy = wrappedDelta(predator.y, target.y, worldHeight);
      const chaseLen = Math.hypot(dx, dy);
      if (chaseLen > 0.0001) {
        chaseX = dx / chaseLen;
        chaseY = dy / chaseLen;
      }
    }

    let separateX = 0;
    let separateY = 0;
    let separateCount = 0;
    for (const other of predators) {
      if (other === predator) {
        continue;
      }
      const dx = wrappedDelta(predator.x, other.x, worldWidth);
      const dy = wrappedDelta(predator.y, other.y, worldHeight);
      const distSq = dx * dx + dy * dy;
      if (distSq <= 0 || distSq > predatorSeparationSq) {
        continue;
      }
      const invDist = 1 / Math.sqrt(distSq);
      separateX -= dx * invDist;
      separateY -= dy * invDist;
      separateCount += 1;
    }
    if (separateCount > 0) {
      separateX /= separateCount;
      separateY /= separateCount;
    }

    const desiredX = chaseX + separateX * config.predatorSeparationWeight;
    const desiredY = chaseY + separateY * config.predatorSeparationWeight;
    if (Math.hypot(desiredX, desiredY) > 0.0001) {
      predator.heading = Math.atan2(desiredY, desiredX);
    }

    predator.vx = Math.cos(predator.heading) * predatorSpeed;
    predator.vy = Math.sin(predator.heading) * predatorSpeed;
    predator.x = wrapCoordinate(predator.x + predator.vx * frameScale, worldWidth);
    predator.y = wrapCoordinate(predator.y + predator.vy * frameScale, worldHeight);

    if (target) {
      const chaseDx = wrappedDelta(predator.x, target.x, worldWidth);
      const chaseDy = wrappedDelta(predator.y, target.y, worldHeight);
      if (chaseDx * chaseDx + chaseDy * chaseDy <= catchSq) {
        caughtIds.add(target.id);
        predator.pauseTimer = rand(
          config.predatorPostCatchPauseMinSeconds,
          config.predatorPostCatchPauseMaxSeconds
        );
        predator.pauseSlowdownTimer = Math.min(config.predatorPauseSlowdownSeconds, predator.pauseTimer);
        predator.pauseStartVx = predator.vx;
        predator.pauseStartVy = predator.vy;
        predator.targetId = null;
        predator.attentionTimer = 0;
      }
    }
  }

  if (caughtIds.size > 0) {
    boids = boids.filter((boid) => !caughtIds.has(boid.id));
    while (boids.length < BOID_COUNT) {
      boids.push(makeBoid());
    }
  }
}

function update(dtSeconds) {
  updateBoids(dtSeconds);
  updatePredators(dtSeconds);
}

function draw() {
  ctx.clearRect(0, 0, worldWidth, worldHeight);

  const gradient = ctx.createLinearGradient(0, 0, 0, worldHeight);
  gradient.addColorStop(0, '#071022');
  gradient.addColorStop(1, '#0a1730');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, worldWidth, worldHeight);

  for (const boid of boids) {
    const angle = boid.heading;
    const head = 5.6;
    const wing = 3.4;

    const x1 = boid.x + Math.cos(angle) * head;
    const y1 = boid.y + Math.sin(angle) * head;
    const x2 = boid.x + Math.cos(angle + 2.45) * wing;
    const y2 = boid.y + Math.sin(angle + 2.45) * wing;
    const x3 = boid.x + Math.cos(angle - 2.45) * wing;
    const y3 = boid.y + Math.sin(angle - 2.45) * wing;

    const speedRatio = Math.hypot(boid.vx, boid.vy) / config.maxSpeed;
    const hue = 185 + speedRatio * 35;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.closePath();
    ctx.fillStyle = `hsl(${hue} 95% 72%)`;
    ctx.fill();
  }

  for (const predator of predators) {
    const angle = predator.heading;
    const head = 9.5;
    const wing = 5.5;

    const x1 = predator.x + Math.cos(angle) * head;
    const y1 = predator.y + Math.sin(angle) * head;
    const x2 = predator.x + Math.cos(angle + 2.35) * wing;
    const y2 = predator.y + Math.sin(angle + 2.35) * wing;
    const x3 = predator.x + Math.cos(angle - 2.35) * wing;
    const y3 = predator.y + Math.sin(angle - 2.35) * wing;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.closePath();
    ctx.fillStyle = '#ff4f5f';
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 170, 170, 0.7)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function loop(timestamp) {
  if (lastFrameTime === 0) {
    lastFrameTime = timestamp;
  }

  const rawDelta = (timestamp - lastFrameTime) / 1000;
  const clampedDelta = Math.min(rawDelta, 0.05);
  lastFrameTime = timestamp;

  update(clampedDelta || FIXED_STEP);
  draw();
  requestAnimationFrame(loop);
}

function renderGameToText() {
  const boidSample = boids.slice(0, 8).map((boid) => ({
    x: Number(boid.x.toFixed(1)),
    y: Number(boid.y.toFixed(1)),
    vx: Number(boid.vx.toFixed(2)),
    vy: Number(boid.vy.toFixed(2)),
    heading: Number((boid.heading * 180 / Math.PI).toFixed(1)),
  }));

  const predatorSample = predators.map((predator) => ({
    x: Number(predator.x.toFixed(1)),
    y: Number(predator.y.toFixed(1)),
    heading: Number((predator.heading * 180 / Math.PI).toFixed(1)),
    targetId: predator.targetId,
    attentionRemaining: Number(predator.attentionTimer.toFixed(2)),
    pauseRemaining: Number(predator.pauseTimer.toFixed(2)),
    pauseSlowdownRemaining: Number(predator.pauseSlowdownTimer.toFixed(2)),
  }));

  return JSON.stringify({
    coordinateSystem: 'origin top-left, +x right, +y down',
    viewport: { width: worldWidth, height: worldHeight },
    boidCount: boids.length,
    predatorCount: predators.length,
    maxTurnAccelerationDegPerSec2: Number(turnAccelRange.value),
    minSpeed: Number(minSpeedRange.value),
    predatorAttentionSeconds: Number(predatorAttentionRange.value),
    boidSample,
    predatorSample,
  });
}

function advanceTime(ms) {
  const totalSeconds = Math.max(0, ms / 1000);
  const fullSteps = Math.floor(totalSeconds / FIXED_STEP);
  const remainder = totalSeconds - fullSteps * FIXED_STEP;

  for (let i = 0; i < fullSteps; i += 1) {
    update(FIXED_STEP);
  }
  if (remainder > 0) {
    update(remainder);
  }

  draw();
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
  for (const predator of predators) {
    predator.attentionTimer = Math.min(predator.attentionTimer, config.predatorAttentionSeconds);
  }
});

restartButton.addEventListener('click', () => {
  initBoids(BOID_COUNT);
  initPredators(PREDATOR_COUNT);
});

window.addEventListener('resize', resizeCanvas);

resizeCanvas();
turnAccelValue.textContent = turnAccelRange.value;
minSpeedValue.textContent = Number(minSpeedRange.value).toFixed(2);
predatorAttentionValue.textContent = Number(predatorAttentionRange.value).toFixed(1);
initBoids(BOID_COUNT);
initPredators(PREDATOR_COUNT);
window.render_game_to_text = renderGameToText;
window.advanceTime = advanceTime;
requestAnimationFrame(loop);
