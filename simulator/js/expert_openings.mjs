// Training-only opening controls. Never alter the game's action masks or RNG.
export function expertOpeningOptions(match) {
  if (!match.isSolo || match.phase !== 'INFO_CHECK') {
    throw Error('Balanced expert openings require a solo briefing');
  }
  const groups = match.bondChoice?.picked === null ? match.bondChoice.groups : null;
  const choices = groups ? groups.map((_, i) => i) : [null];
  return choices.map((bondGroup) => {
    const off = new Set([
      ...match.disabledBonds,
      ...match.staticInactiveBonds,
      ...(groups ? groups.flatMap((g, i) => i === bondGroup ? [] : g) : []),
    ]);
    return {
      bondGroup,
      bandIds: match.gd.bandIds().filter((id) => match.gd.bandBondIds(id).every((b) => !off.has(b))),
    };
  });
}

export function validateExpertOpening(match, opening) {
  if (opening === null || opening === undefined) return;
  const allowed = expertOpeningOptions(match).find((c) => c.bondGroup === opening.bondGroup);
  if (!allowed || !allowed.bandIds.includes(opening.bandId)) {
    throw Error('Expert opening is unavailable or incompatible with its bond group');
  }
}
