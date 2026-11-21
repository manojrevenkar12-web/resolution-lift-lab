/** Motion constants (seconds). One orchestrated reveal on load; nothing else moves by itself. */
export const motion = {
  base: 0.5,
  stagger: 0.18,
  ease: [0.22, 0.9, 0.3, 1] as const,
};
