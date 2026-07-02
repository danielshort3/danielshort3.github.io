module.exports = {
  id: 'classic',
  name: 'Lucky Lambda Classic',
  tier: 'Demo Suite',
  rows: 5,
  reels: 5,
  baseRows: 3,
  baseReels: 3,
  maxRows: 5,
  maxReels: 5,
  symbols: [
    { key: 'cherry', label: 'Cherry' },
    { key: 'lemon', label: 'Lemon' },
    { key: 'orange', label: 'Orange' },
    { key: 'plum', label: 'Plum' },
    { key: 'watermelon', label: 'Watermelon' },
    { key: 'horseshoe', label: 'Horseshoe' },
    { key: 'bell', label: 'Bell' },
    { key: 'diamond', label: 'Diamond' },
    { key: 'seven', label: 'Lucky Seven' },
    { key: 'crown', label: 'Crown' },
    { key: 'wild', label: 'Wild' },
    { key: 'bonus', label: 'Bonus Sigil' }
  ],
  payouts: {
    cherry: 5,
    lemon: 6,
    orange: 7,
    plum: 8,
    watermelon: 10,
    horseshoe: 12,
    bell: 16,
    diamond: 20,
    seven: 25,
    crown: 30,
    wild: 50,
    bonus: 70
  },
  upgradeCosts: {
    rows: [500, 1500],
    reels: [750, 2000],
    lines: [300, 900, 1800]
  }
};
