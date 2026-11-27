import '../simulator/js/context.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';

const { default: kits } = await import('../simulator/engine/server/sim/content/kits/ops/op-haruka.js');

function setup(onDamage) {
  const unit = { id: 1, x: 0, y: 0, skill: { id: 'skchr_haruka_2', active: true } };
  const kit = kits.char_4202_haruka({}, {
    skill: { skillId: unit.skill.id, bb: { atk_scale_extra: 1.5, max_target_extra: 2 } },
  });
  let onHeal;
  const battle = {
    on(name, fn) { assert.equal(name, 'heal'); onHeal = fn; },
    foesInRadius() { return [{ x: 1, y: 0, spawnSeq: 1 }, { x: 2, y: 0, spawnSeq: 2 }]; },
    dealDamage(source, target, damage) { onDamage({ source, target, damage, onHeal, unit }); },
    fx() {},
  };
  kit.talents[1].install(battle, unit);
  return { unit, heal: (amount, opts) => onHeal({ source: unit, target: unit, amount, opts }) };
}

test('Haruka S2 permits lifesteal but cannot retrigger its own splash from that heal', () => {
  const hits = [];
  let healing = 0;
  const { heal } = setup(({ source, damage, onHeal, unit }) => {
    hits.push(damage.amount);
    assert.ok(hits.length <= 4, 'recursive heal-to-damage loop');
    const amount = damage.amount * 0.2;
    onHeal({ source, target: unit, amount, opts: { self: true } });
    healing += amount;
  });
  heal(100);
  assert.deepEqual(hits, [150, 150]);
  assert.equal(healing, 60);
  heal(40); // A later independent heal must still trigger both hits.
  assert.deepEqual(hits, [150, 150, 60, 60]);
  assert.equal(healing, 84);
  heal(100, { regen: true });
  assert.equal(hits.length, 4);
});

test('Haruka S2 releases the recursion guard if damage resolution throws', () => {
  let broken = true, hits = 0;
  const { heal } = setup(() => {
    if (broken) throw Error('injected damage failure');
    hits++;
  });
  assert.throws(() => heal(100), /injected damage failure/);
  broken = false;
  heal(100);
  assert.equal(hits, 2);
});
