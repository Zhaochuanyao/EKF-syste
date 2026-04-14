"""
冒烟测试 - 检查依赖、权重、配置，并用伪造数据跑通完整流程
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_imports():
    print("\n[1/5] 检查 Python 依赖...")
    required = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("cv2", "opencv-python"),
        ("yaml", "pyyaml"),
        ("pydantic", "pydantic"),
    ]
    optional = [
        ("ultralytics", "ultralytics"),
        ("onnxruntime", "onnxruntime"),
        ("fastapi", "fastapi"),
        ("pandas", "pandas"),
    ]
    all_ok = True
    for mod, pkg in required:
        try:
            __import__(mod)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} - 请运行: pip install {pkg}")
            all_ok = False
    for mod, pkg in optional:
        try:
            __import__(mod)
            print(f"  [OK] {pkg} (可选)")
        except ImportError:
            print(f"  [WARN] {pkg} (可选，未安装)")
    return all_ok


def check_weights():
    print("\n[2/5] 检查权重文件...")
    weights = Path("weights/yolov8n.pt")
    if weights.exists():
        print(f"  [OK] {weights}")
        return True
    else:
        print(f"  [WARN] {weights} 不存在")
        print("  提示: 运行 python scripts/download_weights.py 下载权重")
        return False


def check_configs():
    print("\n[3/5] 检查配置文件...")
    configs = [
        "configs/base.yaml",
        "configs/exp/demo_cpu.yaml",
        "configs/tracker/ekf_ctrv.yaml",
    ]
    all_ok = True
    for cfg in configs:
        p = Path(cfg)
        if p.exists():
            print(f"  [OK] {cfg}")
        else:
            print(f"  [FAIL] {cfg} 不存在")
            all_ok = False
    return all_ok


def run_smoke_pipeline():
    print("\n[4/5] 运行伪造数据流程测试...")
    try:
        import numpy as np
        from src.ekf_mot.core.types import Detection
        from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
        from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor
        from src.ekf_mot.tracking.track import Track

        Track.reset_id_counter()
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)
        predictor = TrajectoryPredictor(future_steps=[1, 5, 10])

        # 模拟 10 帧，每帧 2 个目标
        for frame_id in range(10):
            dets = [
                Detection(
                    bbox=np.array([100 + frame_id * 5, 100, 150 + frame_id * 5, 150]),
                    score=0.9, class_id=0, class_name="person",
                ),
                Detection(
                    bbox=np.array([300 + frame_id * 3, 200, 360 + frame_id * 3, 260]),
                    score=0.85, class_id=2, class_name="car",
                ),
            ]
            active = tracker.step(dets, frame_id)

        confirmed = tracker.get_confirmed_tracks()
        print(f"  [OK] 10帧后确认轨迹数: {len(confirmed)}")

        for track in confirmed:
            future = predictor.predict_track(track)
            assert len(future) == 3, f"预测步数应为3，实际为{len(future)}"
        print(f"  [OK] 轨迹预测正常 (steps={list(future.keys())})")

        return True
    except Exception as e:
        print(f"  [FAIL] 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ekf_core():
    print("\n[5/5] 测试 EKF 核心...")
    try:
        import numpy as np
        from src.ekf_mot.filtering.ekf import ExtendedKalmanFilter
        from src.ekf_mot.core.types import Measurement

        ekf = ExtendedKalmanFilter(dt=0.04)
        z0 = np.array([320.0, 240.0, 80.0, 60.0])
        ekf.initialize(z0)

        state = ekf.predict()
        assert state.x.shape == (7,), "状态向量维度错误"
        assert state.P.shape == (7, 7), "协方差矩阵维度错误"

        meas = Measurement(z=np.array([325.0, 242.0, 82.0, 61.0]), score=0.9)
        state = ekf.update(meas)
        assert state.x.shape == (7,)

        # 测试 predict_n_steps 不修改内部状态
        x_before = ekf.x.copy()
        ekf.predict_n_steps(10)
        assert np.allclose(ekf.x, x_before), "predict_n_steps 不应修改内部状态"

        print("  [OK] EKF predict/update 维度正确")
        print("  [OK] predict_n_steps 不修改内部状态")
        return True
    except Exception as e:
        print(f"  [FAIL] EKF 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("EKF 多目标跟踪系统 - 冒烟测试")
    print("=" * 60)

    results = [
        check_imports(),
        check_weights(),
        check_configs(),
        run_smoke_pipeline(),
        check_ekf_core(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    if passed == total:
        print("所有测试通过！系统可以正常运行。")
    else:
        print("部分测试未通过，请根据提示修复问题。")
    print("=" * 60)
    sys.exit(0 if passed >= 3 else 1)
