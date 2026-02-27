const BASE = '/api/v1';

export async function createSimulation(config = {}) {
  const res = await fetch(`${BASE}/simulations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`createSimulation failed: ${res.status}`);
  return res.json();
}

export async function getSimulation(simId) {
  const res = await fetch(`${BASE}/simulations/${simId}`);
  if (!res.ok) throw new Error(`getSimulation failed: ${res.status}`);
  return res.json();
}

export async function getTableGeometry(simId) {
  const res = await fetch(`${BASE}/simulations/${simId}/table_geometry`);
  if (!res.ok) throw new Error(`getTableGeometry failed: ${res.status}`);
  return res.json();
}

export async function strike(simId, params) {
  const res = await fetch(`${BASE}/simulations/${simId}/strikes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`strike failed: ${res.status}`);
  return res.json();
}

export async function getState(simId, t, opts = {}) {
  const params = new URLSearchParams({ t: String(t) });
  if (opts.balls) params.set('balls', opts.balls);
  if (opts.include) params.set('include', opts.include);
  const res = await fetch(`${BASE}/simulations/${simId}/state?${params}`);
  if (!res.ok) throw new Error(`getState failed: ${res.status}`);
  return res.json();
}

export async function getEvents(simId, filters = {}) {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v != null) params.set(k, String(v));
  }
  const qs = params.toString();
  const res = await fetch(`${BASE}/simulations/${simId}/events${qs ? '?' + qs : ''}`);
  if (!res.ok) throw new Error(`getEvents failed: ${res.status}`);
  return res.json();
}

export async function resetSimulation(simId, body = {}) {
  const res = await fetch(`${BASE}/simulations/${simId}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`resetSimulation failed: ${res.status}`);
  return res.json();
}
