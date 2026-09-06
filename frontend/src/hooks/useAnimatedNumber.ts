import { useState, useEffect, useRef } from 'react';

export function useAnimatedNumber(value: number, duration: number = 500, formatter: (v: number) => string = (v) => v.toFixed(1)) {
  const [displayValue, setDisplayValue] = useState(value);
  const startValueRef = useRef(value);
  const endValueRef = useRef(value);
  const startTimeRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (value === endValueRef.current) return;

    startValueRef.current = displayValue;
    endValueRef.current = value;
    startTimeRef.current = performance.now();

    const animate = (time: number) => {
      const elapsed = time - (startTimeRef.current as number);
      const progress = Math.min(elapsed / duration, 1);
      
      // ease-out cubic
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      
      const current = startValueRef.current + (endValueRef.current - startValueRef.current) * easeProgress;
      setDisplayValue(current);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(endValueRef.current);
      }
    };

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value, duration, displayValue]);

  return formatter(displayValue);
}
