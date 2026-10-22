import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import gsap from 'gsap';
import * as THREE from 'three';
import { useUiStore } from '../../stores/uiStore';

export function CameraController() {
  const { camera, controls, scene } = useThree();
  const selectedRackId = useUiStore((state) => state.selectedRackId);
  const perspective = useUiStore((state) => state.perspective);

  useEffect(() => {
    if (selectedRackId) {
      let targetNode = null;
      scene.traverse((child) => {
        if (child.userData && child.userData.rackId === selectedRackId) {
          targetNode = child;
        }
      });

      if (targetNode) {
        const pos = new THREE.Vector3();
        (targetNode as THREE.Object3D).getWorldPosition(pos);
        
        gsap.to(camera.position, {
          x: pos.x + 6,
          y: pos.y + 7,
          z: pos.z + 8,
          duration: 1.5,
          ease: 'power3.inOut',
        });

        if (controls && (controls as any).target) {
          gsap.to((controls as any).target, {
            x: pos.x,
            y: pos.y + 1.5,
            z: pos.z,
            duration: 1.5,
            ease: 'power3.inOut',
          });
        }
      }
    } else {
      // 3D Hero Perspective: Perfectly matches reference image isometric framing
      if (perspective === '3D') {
        gsap.to(camera.position, {
          x: 22.5,
          y: 27.0,
          z: 22.5,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      } else if (perspective === 'TOP') {
        gsap.to(camera.position, {
          x: 0,
          y: 42,
          z: 0.1,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      } else if (perspective === 'SIDE') {
        gsap.to(camera.position, {
          x: 32,
          y: 4,
          z: 0,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      }

      if (controls && (controls as any).target) {
        gsap.to((controls as any).target, {
          x: 0,
          y: 1.0,
          z: 0,
          duration: 1.5,
          ease: 'power3.inOut',
        });
      }
    }
  }, [selectedRackId, perspective, camera, controls, scene]);

  return null;
}
