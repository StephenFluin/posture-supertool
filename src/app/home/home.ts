import {
  Component,
  signal,
  effect,
  ElementRef,
  viewChild,
  ChangeDetectionStrategy,
} from '@angular/core';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

@Component({
  selector: 'app-home',
  imports: [],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="container">
      <h1>Posture & Gait Analysis</h1>

      <div class="controls">
        @if (!isStarted()) {
        <button (click)="startCamera()">Start Analysis</button>
        } @else {
        <button (click)="stopCamera()">Stop</button>
        }
      </div>

      @if (error()) {
      <div class="error">{{ error() }}</div>
      } @if (loading()) {
      <div class="loading">Loading pose detection model...</div>
      }

      <div class="video-container">
        <video #videoElement autoplay playsinline></video>
        <canvas #canvasElement></canvas>
      </div>

      @if (isStarted()) {
      <div class="info">
        <p>FPS: {{ fps() }}</p>
        <p>Keypoints detected: {{ keypointsCount() }}</p>
      </div>
      }
    </div>
  `,
  styles: `
    .container {
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }

    h1 {
      text-align: center;
      margin-bottom: 2rem;
      color: #333;
    }

    .controls {
      display: flex;
      justify-content: center;
      gap: 1rem;
      margin-bottom: 2rem;
    }

    button {
      padding: 0.75rem 1.5rem;
      font-size: 1rem;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.3s;
    }

    button:hover {
      background: #0056b3;
    }

    .error {
      background: #f8d7da;
      color: #721c24;
      padding: 1rem;
      border-radius: 4px;
      margin-bottom: 1rem;
      text-align: center;
    }

    .loading {
      text-align: center;
      padding: 1rem;
      color: #666;
    }

    .video-container {
      position: relative;
      display: flex;
      justify-content: center;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      max-width: 100%;
      margin: 0 auto;
    }

    video {
      width: 100%;
      max-width: 640px;
      display: block;
    }

    canvas {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }

    .info {
      margin-top: 1rem;
      padding: 1rem;
      background: #f8f9fa;
      border-radius: 4px;
      display: flex;
      gap: 2rem;
      justify-content: center;
    }

    .info p {
      margin: 0;
      font-weight: 500;
    }
  `,
})
export class Home {
  private videoElement = viewChild.required<ElementRef<HTMLVideoElement>>('videoElement');
  private canvasElement = viewChild.required<ElementRef<HTMLCanvasElement>>('canvasElement');

  isStarted = signal(false);
  loading = signal(false);
  error = signal('');
  fps = signal(0);
  keypointsCount = signal(0);

  private detector: poseDetection.PoseDetector | null = null;
  private stream: MediaStream | null = null;
  private animationId: number | null = null;
  private lastFrameTime = 0;

  constructor() {
    // Cleanup on destroy
    effect((onCleanup) => {
      onCleanup(() => {
        this.cleanup();
      });
    });
  }

  async startCamera() {
    try {
      this.error.set('');
      this.loading.set(true);

      // Request camera access
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      const video = this.videoElement().nativeElement;
      video.srcObject = this.stream;

      // Wait for video to be ready
      await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          video.play();
          resolve(null);
        };
      });

      // Setup canvas
      const canvas = this.canvasElement().nativeElement;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Initialize pose detector
      await this.initPoseDetector();

      this.isStarted.set(true);
      this.loading.set(false);

      // Start detection loop
      this.detectPose();
    } catch (err) {
      this.error.set(
        `Failed to start camera: ${err instanceof Error ? err.message : 'Unknown error'}`
      );
      this.loading.set(false);
      this.cleanup();
    }
  }

  stopCamera() {
    this.cleanup();
    this.isStarted.set(false);
  }

  private async initPoseDetector() {
    // Ensure TensorFlow.js backend is ready
    await tf.ready();
    
    const model = poseDetection.SupportedModels.MoveNet;
    const detectorConfig: poseDetection.MoveNetModelConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    };
    this.detector = await poseDetection.createDetector(model, detectorConfig);
  }

  private async detectPose() {
    if (!this.isStarted()) return;

    const video = this.videoElement().nativeElement;
    const canvas = this.canvasElement().nativeElement;
    const ctx = canvas.getContext('2d');

    if (!this.detector || !ctx) return;

    try {
      // Detect poses
      const poses = await this.detector.estimatePoses(video);

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw skeleton for each detected pose
      if (poses.length > 0) {
        this.keypointsCount.set(poses[0].keypoints.length);
        this.drawSkeleton(ctx, poses[0]);
      } else {
        this.keypointsCount.set(0);
      }

      // Calculate FPS
      const now = performance.now();
      if (this.lastFrameTime) {
        const delta = now - this.lastFrameTime;
        this.fps.set(Math.round(1000 / delta));
      }
      this.lastFrameTime = now;
    } catch (err) {
      console.error('Pose detection error:', err);
    }

    // Continue loop
    this.animationId = requestAnimationFrame(() => this.detectPose());
  }

  private drawSkeleton(ctx: CanvasRenderingContext2D, pose: poseDetection.Pose) {
    const keypoints = pose.keypoints;
    const minConfidence = 0.3;

    // Define skeleton connections
    const connections: [number, number][] = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 4], // Head
      [5, 6],
      [5, 7],
      [7, 9],
      [6, 8],
      [8, 10], // Arms
      [5, 11],
      [6, 12],
      [11, 12], // Torso
      [11, 13],
      [13, 15],
      [12, 14],
      [14, 16], // Legs
    ];

    // Draw connections (bones)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    connections.forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];
      if (kp1.score && kp2.score && kp1.score > minConfidence && kp2.score > minConfidence) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }
    });

    // Draw keypoints (joints)
    keypoints.forEach((keypoint) => {
      if (keypoint.score && keypoint.score > minConfidence) {
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  private cleanup() {
    // Stop animation loop
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Stop video stream
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    // Clear video element
    const video = this.videoElement()?.nativeElement;
    if (video) {
      video.srcObject = null;
    }

    // Clear canvas
    const canvas = this.canvasElement()?.nativeElement;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }

    this.lastFrameTime = 0;
  }
}
