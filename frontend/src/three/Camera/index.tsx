import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import gsap from 'gsap';
import * as THREE from 'three';
import { useUiStore } from '../../stores/uiStore';

export function CameraController() {
  const { camera, controls, scene } = useThree();
  const selectedRackId = useUiStore((state) => state.selectedRackId);

  useEffect(() => {
    if (selectedRackId) {
      // Find the specific rack node in the scene
      let targetNode = null;
      scene.traverse((child) => {
        if (child.userData && child.userData.rackId === selectedRackId) {
          targetNode = child;
        }
      });

      if (targetNode) {
        const pos = new THREE.Vector3();
        (targetNode as THREE.Object3D).getWorldPosition(pos);
        
        // Offset the camera to look AT the rack from a good angle
        gsap.to(camera.position, {
          x: pos.x + 8,
          y: pos.y + 10,
          z: pos.z + 12,
          duration: 1.5,
          ease: 'power3.inOut',
        });

        if (controls && (controls as any).target) {
          gsap.to((controls as any).target, {
            x: pos.x,
            y: pos.y + 3,
            z: pos.z,
            duration: 1.5,
            ease: 'power3.inOut',
          });
        }
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
