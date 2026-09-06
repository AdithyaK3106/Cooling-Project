import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import gsap from 'gsap';
import { useUiStore } from '../../stores/uiStore';

export function CameraController() {
  const { camera, controls } = useThree();
  const selectedRackId = useUiStore((state) => state.selectedRackId);

  useEffect(() => {
    if (selectedRackId) {
      // For now, hardcode a focus position or use a dummy position
      // In Phase 7, we'll map this to the exact rack's world position
      gsap.to(camera.position, {
        x: 10,
        y: 15,
        z: 10,
        duration: 1.5,
        ease: 'power3.inOut',
      });

      if (controls && (controls as any).target) {
        gsap.to((controls as any).target, {
          x: 0,
          y: 5,
          z: 0,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      }
    } else {
      // Reset to isometric overview
      gsap.to(camera.position, {
        x: 50,
        y: 50,
        z: 50,
        duration: 1.5,
        ease: 'power3.inOut',
      });

      if (controls && (controls as any).target) {
        gsap.to((controls as any).target, {
          x: 0,
          y: 0,
          z: 0,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      }
    }
  }, [selectedRackId, camera, controls]);

  return null;
}
