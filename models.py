"""
Database models for Attendify.

User             - authenticated user (teacher/admin or student)
Student          - a registered person with multiple face embeddings and MAC address
Course           - academic course/subject (e.g. CS301 Data Structures)
TimetableSlot    - recurring weekly lecture schedule
AttendanceSession- an active attendance time window created by a teacher
AttendanceLog    - one attendance event with manual audit override support
SystemSetting    - dynamic system configuration (e.g., threshold)
"""
import json
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")  # 'teacher', 'admin', 'student'
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship("Student", backref=db.backref("user", uselist=False))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_teacher(self):
        return self.role in ("teacher", "admin")

    @property
    def is_admin(self):
        return self.role == "admin"


class Course(db.Model):
    __tablename__ = "courses"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(30), unique=True, nullable=False)   # e.g. "CS301"
    title = db.Column(db.String(150), nullable=False)              # e.g. "Data Structures & Algorithms"
    department = db.Column(db.String(100), default="Computer Science & Engineering")
    instructor = db.Column(db.String(100), default="Faculty Member")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    timetable_slots = db.relationship("TimetableSlot", backref="course", lazy=True, cascade="all, delete-orphan")
    sessions = db.relationship("AttendanceSession", backref="course_rel", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "title": self.title,
            "department": self.department,
            "instructor": self.instructor,
            "slot_count": len(self.timetable_slots),
        }


class TimetableSlot(db.Model):
    __tablename__ = "timetable_slots"

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=False)
    day_of_week = db.Column(db.String(20), nullable=False)        # "Monday", "Tuesday", etc.
    start_time = db.Column(db.String(5), nullable=False)          # "10:00" (24-hr format)
    end_time = db.Column(db.String(5), nullable=False)            # "11:00"
    room = db.Column(db.String(60), default="Lecture Hall 101")
    is_active = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            "id": self.id,
            "course_id": self.course_id,
            "course_code": self.course.code if self.course else "-",
            "course_title": self.course.title if self.course else "-",
            "day_of_week": self.day_of_week,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "room": self.room,
            "is_active": self.is_active,
        }


class AttendanceSession(db.Model):
    __tablename__ = "attendance_sessions"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    course_code = db.Column(db.String(50), nullable=False, default="GEN101")
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(80), nullable=True)

    attendance_logs = db.relationship("AttendanceLog", backref="session", lazy=True)

    def is_open(self):
        now = datetime.utcnow()
        return self.is_active and (self.start_time <= now <= self.end_time)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "course_code": self.course_code,
            "course_id": self.course_id,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M"),
            "is_open": self.is_open(),
            "is_active": self.is_active,
        }


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    birth_year = db.Column(db.String(4), nullable=True)
    mac_address = db.Column(db.String(17), unique=True, nullable=False)
    photo_path = db.Column(db.String(255), nullable=False, default="student_photos/default.png")
    enrollment_status = db.Column(db.String(30), default="ACTIVE")  # ACTIVE, PENDING_BIOMETRICS

    # Face embedding vectors, stored as JSON list of lists [[512-d], ...]
    embeddings = db.Column(db.Text, nullable=False, default="[]")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendance_logs = db.relationship(
        "AttendanceLog", backref="student", lazy=True,
        cascade="all, delete-orphan"
    )

    def set_embeddings(self, vectors):
        """Accepts a list of numpy vectors or lists and stores as JSON."""
        if vectors is None or len(vectors) == 0:
            self.embeddings = "[]"
            return
        
        processed = []
        for vec in vectors:
            if hasattr(vec, "tolist"):
                processed.append(vec.tolist())
            elif isinstance(vec, (list, tuple)):
                processed.append([float(x) for x in vec])
            else:
                processed.append(list(map(float, vec)))
        
        self.embeddings = json.dumps(processed)

    def get_embeddings(self):
        try:
            data = json.loads(self.embeddings)
            if data and not isinstance(data[0], list):
                return [data]
            return data
        except Exception:
            return []

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "roll_no": self.roll_no,
            "phone_number": self.phone_number or "-",
            "birth_year": self.birth_year or "-",
            "mac_address": self.mac_address,
            "photo_path": self.photo_path,
            "enrollment_status": self.enrollment_status or "ACTIVE",
            "embedding_count": len(self.get_embeddings()),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class AttendanceLog(db.Model):
    __tablename__ = "attendance_logs"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("attendance_sessions.id"), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    face_match_score = db.Column(db.Float, nullable=False, default=1.0)   # similarity, 0-1
    wifi_verified = db.Column(db.Boolean, nullable=False, default=True)
    liveness_verified = db.Column(db.Boolean, default=True)
    matched_mac = db.Column(db.String(17), nullable=True)

    # "PRESENT", "FLAGGED", "EXCUSED_LEAVE", "ABSENT"
    status = db.Column(db.String(20), nullable=False)

    # Manual Override & Academic Exception Audit Fields
    is_manual_override = db.Column(db.Boolean, default=False)
    override_by = db.Column(db.String(80), nullable=True)
    override_reason = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "session_id": self.session_id,
            "session_title": self.session.title if self.session else "Ad-hoc / Open",
            "student_name": self.student.name if self.student else "Unknown",
            "roll_no": self.student.roll_no if self.student else "-",
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "face_match_score": round(self.face_match_score, 4),
            "wifi_verified": self.wifi_verified,
            "liveness_verified": self.liveness_verified,
            "matched_mac": self.matched_mac,
            "status": self.status,
            "is_manual_override": self.is_manual_override or False,
            "override_by": self.override_by,
            "override_reason": self.override_reason,
        }


class SystemSetting(db.Model):
    __tablename__ = "system_settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(255), nullable=False)

    @classmethod
    def get_setting(cls, key, default=None):
        setting = cls.query.filter_by(key=key).first()
        return setting.value if setting else default

    @classmethod
    def set_setting(cls, key, value):
        setting = cls.query.filter_by(key=key).first()
        if not setting:
            setting = cls(key=key, value=str(value))
            db.session.add(setting)
        else:
            setting.value = str(value)
        db.session.commit()
