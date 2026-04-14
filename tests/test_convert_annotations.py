"""
标注格式转换工具测试

覆盖：
  - UA-DETRAC XML 解析（用内存中构造的最小 XML 字符串）
  - MOT17 gt.txt 解析（用临时文件）
  - 边界框坐标转换正确性
  - 非法输入的异常处理
  - 生成 JSON 格式符合内部规范
"""

import json
import pytest
from pathlib import Path

# 最小 UA-DETRAC XML 样本
MINIMAL_UA_DETRAC_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<sequence name="TEST_SEQ">
  <frame density="1" num="0">
    <target_list>
      <target id="1">
        <box left="100.0" top="200.0" width="50.0" height="30.0" />
        <attribute vehicle_type="car" orientation="rearview" speed="10.0"
                   trajectory_length="5" truncation_ratio="0" />
      </target>
      <target id="2">
        <box left="300.0" top="400.0" width="60.0" height="40.0" />
        <attribute vehicle_type="bus" orientation="frontalview" speed="5.0"
                   trajectory_length="3" truncation_ratio="0" />
      </target>
    </target_list>
  </frame>
  <frame density="1" num="1">
    <target_list>
      <target id="1">
        <box left="105.0" top="200.0" width="50.0" height="30.0" />
        <attribute vehicle_type="car" orientation="rearview" speed="10.0"
                   trajectory_length="5" truncation_ratio="0" />
      </target>
    </target_list>
  </frame>
</sequence>
"""

# 最小 MOT17 gt.txt 样本
MINIMAL_MOT17_GT = """\
1,1,100,200,50,30,1,1,0.8
1,2,300,400,60,40,1,1,0.9
2,1,105,200,50,30,1,1,0.8
2,3,0,0,0,0,0,1,0.0
"""
# 第4行 conf=0，应跳过


# ══════════════════════════════════════════════════════════════
# UA-DETRAC 解析测试
# ══════════════════════════════════════════════════════════════

class TestUADETRAC:
    @pytest.fixture
    def xml_file(self, tmp_path):
        """写入最小 XML 到临时文件"""
        p = tmp_path / "TEST_SEQ.xml"
        p.write_text(MINIMAL_UA_DETRAC_XML, encoding="utf-8")
        return p

    def test_parse_sequence_name(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        assert data["sequence"] == "TEST_SEQ"

    def test_parse_dataset_tag(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        assert data["dataset"] == "ua_detrac"

    def test_frame_count(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        assert data["total_frames"] == 2
        assert len(data["frames"]) == 2

    def test_annotation_count_per_frame(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        frame0 = data["frames"][0]
        frame1 = data["frames"][1]
        assert frame0["frame_id"] == 0
        assert len(frame0["annotations"]) == 2
        assert frame1["frame_id"] == 1
        assert len(frame1["annotations"]) == 1

    def test_bbox_conversion(self, xml_file):
        """box left/top/width/height → x1,y1,x2,y2"""
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        ann = data["frames"][0]["annotations"][0]
        # left=100, top=200, width=50, height=30 → x1=100, y1=200, x2=150, y2=230
        assert ann["bbox"] == [100.0, 200.0, 150.0, 230.0]

    def test_class_mapping_car(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        ann = data["frames"][0]["annotations"][0]
        assert ann["class_name"] == "car"
        assert ann["class_id"] == 0

    def test_class_mapping_bus(self, xml_file):
        from scripts.convert_annotations import _parse_ua_detrac_xml
        data = _parse_ua_detrac_xml(xml_file)
        ann = data["frames"][0]["annotations"][1]
        assert ann["class_name"] == "bus"
        assert ann["class_id"] == 1

    def test_convert_ua_detrac_single_file(self, xml_file, tmp_path):
        """convert_ua_detrac 对单文件生成 JSON"""
        from scripts.convert_annotations import convert_ua_detrac
        out_dir = tmp_path / "out"
        generated = convert_ua_detrac(str(xml_file), str(out_dir))
        assert len(generated) == 1
        out_file = generated[0]
        assert out_file.exists()
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["sequence"] == "TEST_SEQ"
        assert data["total_frames"] == 2

    def test_convert_ua_detrac_directory(self, tmp_path):
        """convert_ua_detrac 对目录批量转换"""
        from scripts.convert_annotations import convert_ua_detrac
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()
        # 写入两个 XML 文件
        for name in ["SEQ_A", "SEQ_B"]:
            xml = MINIMAL_UA_DETRAC_XML.replace("TEST_SEQ", name)
            (xml_dir / f"{name}.xml").write_text(xml, encoding="utf-8")
        out_dir = tmp_path / "out"
        generated = convert_ua_detrac(str(xml_dir), str(out_dir))
        assert len(generated) == 2

    def test_invalid_xml_raises(self, tmp_path):
        """无效 XML 应抛出异常"""
        from scripts.convert_annotations import _parse_ua_detrac_xml
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("<unclosed>", encoding="utf-8")
        with pytest.raises(ValueError, match="XML 解析失败"):
            _parse_ua_detrac_xml(bad_file)

    def test_nonexistent_input_raises(self, tmp_path):
        """不存在的路径应抛出 FileNotFoundError"""
        from scripts.convert_annotations import convert_ua_detrac
        with pytest.raises(FileNotFoundError):
            convert_ua_detrac(str(tmp_path / "not_exist.xml"), str(tmp_path / "out"))


# ══════════════════════════════════════════════════════════════
# MOT17 解析测试
# ══════════════════════════════════════════════════════════════

class TestMOT17:
    @pytest.fixture
    def mot17_seq_dir(self, tmp_path):
        """构造最小 MOT17 序列目录"""
        seq_dir = tmp_path / "MOT17-02-FRCNN"
        gt_dir = seq_dir / "gt"
        gt_dir.mkdir(parents=True)
        (gt_dir / "gt.txt").write_text(MINIMAL_MOT17_GT, encoding="utf-8")
        # seqinfo.ini
        (seq_dir / "seqinfo.ini").write_text(
            "[Sequence]\nname=MOT17-02-FRCNN\nframerate=25\nseqlength=300\n",
            encoding="utf-8",
        )
        return seq_dir

    def test_parse_dataset_tag(self, mot17_seq_dir):
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        assert data["dataset"] == "mot17"

    def test_parse_sequence_name(self, mot17_seq_dir):
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        assert data["sequence"] == "MOT17-02-FRCNN"

    def test_fps_from_seqinfo(self, mot17_seq_dir):
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        assert data["fps"] == 25

    def test_frame_count(self, mot17_seq_dir):
        """gt.txt 中有帧1和帧2（忽略 conf=0 的行）"""
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        # 帧1: 2个有效标注, 帧2: 1个有效标注（conf=0被跳过）
        assert data["total_frames"] == 2

    def test_ignore_conf_zero(self, mot17_seq_dir):
        """conf=0 的行（忽略区域）应被跳过"""
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        frame2 = next(f for f in data["frames"] if f["frame_id"] == 2)
        assert len(frame2["annotations"]) == 1  # conf=0 那行被跳过

    def test_bbox_conversion(self, mot17_seq_dir):
        """MOT17 x,y,w,h → x1,y1,x2,y2"""
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        frame1 = next(f for f in data["frames"] if f["frame_id"] == 1)
        ann = next(a for a in frame1["annotations"] if a["id"] == 1)
        # x=100, y=200, w=50, h=30 → x1=100, y1=200, x2=150, y2=230
        assert ann["bbox"] == [100.0, 200.0, 150.0, 230.0]

    def test_class_pedestrian(self, mot17_seq_dir):
        """MOT17 class=1 → pedestrian"""
        from scripts.convert_annotations import _parse_mot17_sequence
        data = _parse_mot17_sequence(mot17_seq_dir)
        ann = data["frames"][0]["annotations"][0]
        assert ann["class_name"] == "pedestrian"
        assert ann["class_id"] == 0

    def test_convert_mot17_single_seq(self, mot17_seq_dir, tmp_path):
        from scripts.convert_annotations import convert_mot17
        out_dir = tmp_path / "out"
        generated = convert_mot17(str(mot17_seq_dir), str(out_dir))
        assert len(generated) == 1
        assert generated[0].exists()

    def test_convert_mot17_directory(self, tmp_path):
        """传入包含多个序列目录的父目录"""
        from scripts.convert_annotations import convert_mot17
        parent = tmp_path / "train"
        # 创建两个序列目录
        for name in ["SEQ_A", "SEQ_B"]:
            d = parent / name / "gt"
            d.mkdir(parents=True)
            (d / "gt.txt").write_text(MINIMAL_MOT17_GT, encoding="utf-8")
        out_dir = tmp_path / "out"
        generated = convert_mot17(str(parent), str(out_dir))
        assert len(generated) == 2

    def test_missing_gt_raises(self, tmp_path):
        """缺少 gt/gt.txt 时应抛出 FileNotFoundError"""
        from scripts.convert_annotations import _parse_mot17_sequence
        seq_dir = tmp_path / "NO_GT"
        seq_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _parse_mot17_sequence(seq_dir)


# ══════════════════════════════════════════════════════════════
# JSON 格式验证
# ══════════════════════════════════════════════════════════════

class TestValidateJSON:
    def test_valid_json_passes(self, tmp_path):
        from scripts.convert_annotations import validate_json
        valid_data = {
            "dataset": "ua_detrac",
            "sequence": "TEST",
            "fps": 25,
            "total_frames": 1,
            "frames": [
                {
                    "frame_id": 0,
                    "annotations": [
                        {"id": 1, "bbox": [0, 0, 100, 100], "class_id": 0, "class_name": "car"}
                    ],
                }
            ],
        }
        p = tmp_path / "valid.json"
        p.write_text(json.dumps(valid_data), encoding="utf-8")
        assert validate_json(p) is True

    def test_missing_field_fails(self, tmp_path):
        from scripts.convert_annotations import validate_json
        bad_data = {"dataset": "ua_detrac", "sequence": "TEST"}  # 缺少 frames, fps
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad_data), encoding="utf-8")
        assert validate_json(p) is False

    def test_invalid_json_fails(self, tmp_path):
        from scripts.convert_annotations import validate_json
        p = tmp_path / "malformed.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert validate_json(p) is False
