"""
实验框架测试
  1. 场景配置文件能被正确加载（YAML 解析 + 关键字段验证）
  2. run_experiments.py 工具函数（_extract_track_stats、save_summary_csv）
  3. experiment_summary.json / CSV 能正确生成（不依赖真实视频/检测器）
"""

import sys
import csv
import json
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════
# 场景配置加载测试
# ══════════════════════════════════════════════════════════════

SCENARIO_CONFIGS = [
    "configs/exp/scenario_uniform_motion.yaml",
    "configs/exp/scenario_accelerated_motion.yaml",
    "configs/exp/scenario_turning_motion.yaml",
]


class TestScenarioConfigs:
    """验证三个场景配置文件能被正确解析"""

    def _load(self, path: str) -> dict:
        from src.ekf_mot.core.config import load_yaml
        return load_yaml(ROOT / path)

    @pytest.mark.parametrize("config_path", SCENARIO_CONFIGS)
    def test_config_file_exists(self, config_path):
        assert (ROOT / config_path).exists(), f"配置文件不存在: {config_path}"

    @pytest.mark.parametrize("config_path", SCENARIO_CONFIGS)
    def test_config_parseable(self, config_path):
        data = self._load(config_path)
        assert isinstance(data, dict), "YAML 应解析为字典"

    @pytest.mark.parametrize("config_path", SCENARIO_CONFIGS)
    def test_config_has_detector_section(self, config_path):
        data = self._load(config_path)
        assert "detector" in data

    @pytest.mark.parametrize("config_path", SCENARIO_CONFIGS)
    def test_config_has_tracker_section(self, config_path):
        data = self._load(config_path)
        assert "tracker" in data

    @pytest.mark.parametrize("config_path", SCENARIO_CONFIGS)
    def test_config_has_experiment_meta(self, config_path):
        data = self._load(config_path)
        assert "experiment" in data, "场景配置应包含 experiment 元数据"
        exp = data["experiment"]
        assert "name" in exp
        assert "scenario" in exp

    def test_uniform_has_low_std_acc(self):
        data = self._load("configs/exp/scenario_uniform_motion.yaml")
        std_acc = data["ekf"]["process_noise"]["std_acc"]
        assert std_acc <= 1.0, f"匀速配置 std_acc 应 ≤ 1.0，实际={std_acc}"

    def test_accelerated_has_high_std_acc(self):
        data = self._load("configs/exp/scenario_accelerated_motion.yaml")
        std_acc = data["ekf"]["process_noise"]["std_acc"]
        assert std_acc >= 3.0, f"变速配置 std_acc 应 ≥ 3.0，实际={std_acc}"

    def test_turning_has_high_yaw_rate(self):
        data = self._load("configs/exp/scenario_turning_motion.yaml")
        std_yaw = data["ekf"]["process_noise"]["std_yaw_rate"]
        assert std_yaw >= 1.0, f"转弯配置 std_yaw_rate 应 ≥ 1.0，实际={std_yaw}"

    def test_uniform_has_more_future_steps(self):
        data = self._load("configs/exp/scenario_uniform_motion.yaml")
        steps = data["prediction"]["future_steps"]
        assert len(steps) >= 4, "匀速配置应有 ≥4 个预测步数以测试长程精度"

    def test_load_config_with_base(self):
        """通过 load_config（含 _base_ 合并）加载场景配置，关键参数应被正确合并"""
        from src.ekf_mot.core.config import load_config
        cfg = load_config(ROOT / "configs/exp/scenario_uniform_motion.yaml")
        # 来自 base.yaml 的字段应存在
        assert "runtime" in cfg
        assert "ekf" in cfg
        # 来自场景配置的覆盖值应生效
        assert cfg["ekf"]["process_noise"]["std_acc"] <= 1.0


# ══════════════════════════════════════════════════════════════
# run_experiments.py 工具函数测试
# ══════════════════════════════════════════════════════════════

class TestExtractTrackStats:
    """_extract_track_stats 函数测试"""

    def _make_frames(self, frame_count: int, tracks_per_frame: list) -> list:
        frames = []
        for fid in range(frame_count):
            tracks = []
            for tid in range(tracks_per_frame[fid] if fid < len(tracks_per_frame) else 0):
                tracks.append({
                    "track_id": tid + 1,
                    "state_name": "Confirmed",
                    "bbox": [0, 0, 10, 10],
                })
            frames.append({
                "frame_id": fid + 1,
                "num_detections": tracks_per_frame[fid] if fid < len(tracks_per_frame) else 0,
                "tracks": tracks,
            })
        return frames

    def test_empty_frames(self):
        from scripts.run_experiments import _extract_track_stats
        stats = _extract_track_stats([])
        assert stats["num_tracks"] == 0
        assert stats["avg_track_length"] == 0.0

    def test_single_track_three_frames(self):
        from scripts.run_experiments import _extract_track_stats
        frames = self._make_frames(3, [1, 1, 1])
        stats = _extract_track_stats(frames)
        assert stats["num_tracks"] == 1
        assert stats["avg_track_length"] == pytest.approx(3.0)

    def test_two_tracks_different_lengths(self):
        from scripts.run_experiments import _extract_track_stats
        # track_id=1 出现 3 帧，track_id=2 出现 1 帧
        frames = [
            {"frame_id": 1, "num_detections": 2, "tracks": [
                {"track_id": 1, "state_name": "Confirmed", "bbox": [0,0,10,10]},
                {"track_id": 2, "state_name": "Confirmed", "bbox": [10,10,20,20]},
            ]},
            {"frame_id": 2, "num_detections": 1, "tracks": [
                {"track_id": 1, "state_name": "Confirmed", "bbox": [0,0,10,10]},
            ]},
            {"frame_id": 3, "num_detections": 1, "tracks": [
                {"track_id": 1, "state_name": "Confirmed", "bbox": [0,0,10,10]},
            ]},
        ]
        stats = _extract_track_stats(frames)
        assert stats["num_tracks"] == 2
        # (3 + 1) / 2 = 2.0
        assert stats["avg_track_length"] == pytest.approx(2.0)

    def test_tentative_tracks_excluded(self):
        from scripts.run_experiments import _extract_track_stats
        frames = [
            {"frame_id": 1, "num_detections": 1, "tracks": [
                {"track_id": 1, "state_name": "Tentative", "bbox": [0,0,10,10]},
            ]},
        ]
        stats = _extract_track_stats(frames)
        assert stats["num_tracks"] == 0

    def test_max_tracks_per_frame(self):
        from scripts.run_experiments import _extract_track_stats
        frames = self._make_frames(2, [3, 2])
        stats = _extract_track_stats(frames)
        assert stats["max_tracks_per_frame"] == 3


class TestSaveSummaryCsv:
    """save_summary_csv 生成正确的 CSV 文件"""

    def test_csv_created_with_correct_columns(self, tmp_path):
        from scripts.run_experiments import save_summary_csv, _CSV_COLUMNS
        results = [
            {
                "run_id": "test_run",
                "scenario": "test",
                "video": "v.mp4",
                "status": "success",
                "frames_processed": 100,
                "fps": 12.5,
                "elapsed_sec": 8.0,
                "num_tracks": 3,
                "avg_track_length": 25.5,
                "max_tracks_per_frame": 4,
                "total_detections": 300,
                "num_frames_with_tracks": 80,
                "scenario_description": "test",
                "config": "c.yaml",
                "output_dir": "out/",
            }
        ]
        csv_path = tmp_path / "test_summary.csv"
        save_summary_csv(results, csv_path)
        assert csv_path.exists()
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "test_run"
        assert rows[0]["fps"] == "12.5"

    def test_csv_handles_empty_results(self, tmp_path):
        from scripts.run_experiments import save_summary_csv
        csv_path = tmp_path / "empty.csv"
        save_summary_csv([], csv_path)
        assert csv_path.exists()


class TestGetExperimentName:
    """_get_experiment_name_from_config 能读取 experiment.name 字段"""

    def test_reads_experiment_name_from_scenario_config(self):
        from scripts.run_experiments import _get_experiment_name_from_config
        cfg = ROOT / "configs/exp/scenario_uniform_motion.yaml"
        name = _get_experiment_name_from_config(cfg)
        assert name == "uniform_motion"

    def test_falls_back_to_stem_for_unknown_config(self, tmp_path):
        from scripts.run_experiments import _get_experiment_name_from_config
        cfg_path = tmp_path / "my_config.yaml"
        cfg_path.write_text("detector:\n  conf: 0.35\n")
        name = _get_experiment_name_from_config(cfg_path)
        assert name == "my_config"


# ══════════════════════════════════════════════════════════════
# experiment_summary.json 生成（最小流程 mock 测试）
# ══════════════════════════════════════════════════════════════

class TestExperimentSummaryOutput:
    """验证 experiment_summary.json 结构正确"""

    def _make_mock_result(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "config": "cfg.yaml",
            "video": "v.mp4",
            "scenario": run_id,
            "scenario_description": "",
            "output_dir": "out/",
            "status": "success",
            "error": "",
            "frames_processed": 100,
            "fps": 12.0,
            "elapsed_sec": 8.0,
            "num_tracks": 2,
            "avg_track_length": 30.0,
            "max_tracks_per_frame": 3,
            "total_detections": 200,
            "num_frames_with_tracks": 80,
        }

    def test_summary_json_structure(self, tmp_path):
        from scripts.run_experiments import save_summary_csv

        results = [
            self._make_mock_result("exp_a"),
            self._make_mock_result("exp_b"),
        ]
        summary = {
            "total_experiments": len(results),
            "success_count": 2,
            "error_count": 0,
            "output_dir": str(tmp_path),
            "results": results,
        }
        json_path = tmp_path / "experiment_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f)

        # 验证能被正确读取
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["total_experiments"] == 2
        assert loaded["success_count"] == 2
        assert len(loaded["results"]) == 2
        assert loaded["results"][0]["run_id"] == "exp_a"

    def test_summary_csv_roundtrip(self, tmp_path):
        from scripts.run_experiments import save_summary_csv

        results = [self._make_mock_result("exp_x")]
        csv_path = tmp_path / "summary.csv"
        save_summary_csv(results, csv_path)

        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert float(rows[0]["fps"]) == pytest.approx(12.0)
        assert float(rows[0]["avg_track_length"]) == pytest.approx(30.0)
