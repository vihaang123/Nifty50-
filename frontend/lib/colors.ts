// Behaviour-class colours match the Python charts. Cap colours are for the PCA scatter only.
// Two of these have low contrast on white (Balanced green, Mid-cap magenta), so every chart that uses them also shows a legend.
export const BEHAVIOR_COLORS: Record<string, string> = {
  Defensive: "#2a78d6",
  Balanced: "#1baf7a",
  Aggressive: "#eb6834",
};
export const CAP_COLORS: Record<string, string> = {
  "Large Cap": "#4a3aa7",
  "Mid Cap": "#e87ba4",
  "Small Cap": "#008300",
};
export const FALLBACK_COLOR = "#8a94a0";
export const behaviorColor = (name: string | null | undefined) => (name && BEHAVIOR_COLORS[name]) || FALLBACK_COLOR;
export const capColor = (name: string | null | undefined) => (name && CAP_COLORS[name]) || FALLBACK_COLOR;
export const BEHAVIOR_ORDER = ["Defensive", "Balanced", "Aggressive"];
export const CAP_ORDER = ["Large Cap", "Mid Cap", "Small Cap"];
