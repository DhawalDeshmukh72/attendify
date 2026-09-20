"""
Attendify — Smart Enterprise Attendance System
Face recognition (DeepFace/Facenet512) + FAISS Vector Search + Liveness
Verification + Dynamic Time-Windows + Role-Based Access (Teacher/Admin vs Student).
"""
import csv
import io
import os
import re
from datetime import datetime, date, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify,
    Response, send_from_directory
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from models import (
    db, User, Student, Course, TimetableSlot, AttendanceSession, AttendanceLog, SystemSetting
)
import face_utils
import wifi_simulator
from faiss_index import vector_index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "data", "student_photos")
DB_PATH = os.path.join(BASE_DIR, "data", "attendify.db")

os.makedirs(PHOTO_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "attendify-enterprise-secret-key-2026")

# Dynamic Database Configuration (PostgreSQL / SQLite with Pooling)
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

if not DATABASE_URL.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 10,
        "max_overflow": 20,
    }

# Production Session Cookie Hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

db.init_app(app)

# CSRF Protection
csrf = CSRFProtect(app)

# Rate Limiter Configuration
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per hour", "200 per minute"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message_category = "warning"
login_manager.init_app(app)

MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def teacher_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_teacher:
            flash("Access denied. Teacher/Admin role required.", "danger")
            return redirect(url_for("student_portal"))
        return f(*args, **kwargs)
    return decorated_function


def rebuild_vector_index():
    """Rebuild in-memory FAISS index from SQLite database."""
    with app.app_context():
        vector_index.clear()
        students = Student.query.all()
        for s in students:
            embeddings = s.get_embeddings()
            if embeddings:
                vector_index.add(embeddings, s.id)


with app.app_context():
    db.create_all()

    # Safe dynamic column migration
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE students ADD COLUMN photo_base64 TEXT"))
            conn.commit()
    except Exception:
        pass

    # Seed default admin / teacher account if none exists
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", role="admin")
        admin.set_password("admin123")
        db.session.add(admin)

    # Seed default teacher account
    if not User.query.filter_by(username="teacher").first():
        teacher = User(username="teacher", role="teacher")
        teacher.set_password("teacher123")
        db.session.add(teacher)

    # Seed default academic courses
    if Course.query.count() == 0:
        c1 = Course(code="CS301", title="Data Structures & Algorithms", department="Computer Science", instructor="Prof. Alan Turing")
        c2 = Course(code="CS402", title="Machine Learning & Neural Networks", department="Computer Science", instructor="Dr. Geoffrey Hinton")
        c3 = Course(code="CS204", title="Database Management Systems", department="Information Technology", instructor="Dr. Edgar Codd")
        db.session.add_all([c1, c2, c3])
        db.session.flush()

        # Seed sample weekly timetable slots
        s1 = TimetableSlot(course_id=c1.id, day_of_week="Monday", start_time="10:00", end_time="11:00", room="Auditorium Hall 101")
        s2 = TimetableSlot(course_id=c2.id, day_of_week="Wednesday", start_time="11:30", end_time="12:30", room="AI & Robotics Lab 302")
        s3 = TimetableSlot(course_id=c3.id, day_of_week="Friday", start_time="09:00", end_time="10:00", room="Lecture Theater 204")
        db.session.add_all([s1, s2, s3])

    # Seed default match threshold setting if not set
    if not SystemSetting.get_setting("match_threshold"):
        SystemSetting.set_setting("match_threshold", "0.68")

    # Seed default student & user account if none exists
    if Student.query.count() == 0:
        default_student = Student(
            roll_no="1432231043",
            name="Dhawal Nisarg Deshmukh",
            birth_year=2003,
            phone_number="8767454366",
            mac_address="00:1B:44:11:3A:B7",
            photo_path="student_photos/default.png"
        )
        db.session.add(default_student)
        db.session.flush()

        # Seed student login account
        if not User.query.filter_by(username="1432231043").first():
            student_user = User(username="1432231043", role="student")
            student_user.set_password("student123")
            db.session.add(student_user)

        # Seed an active session for immediate testing
        if AttendanceSession.query.count() == 0:
            active_course = Course.query.first()
            if active_course:
                sample_session = AttendanceSession(
                    title="GenAI & Prompt Engineering - Lecture 01",
                    course_code=active_course.code,
                    course_id=active_course.id,
                    start_time=datetime.utcnow() - timedelta(minutes=15),
                    end_time=datetime.utcnow() + timedelta(hours=2),
                    is_active=True,
                    created_by="admin"
                )
                db.session.add(sample_session)
                db.session.flush()

                # Add sample attendance log
                sample_log = AttendanceLog(
                    student_id=default_student.id,
                    session_id=sample_session.id,
                    status="PRESENT",
                    face_match_score=0.92,
                    wifi_verified=True,
                    liveness_verified=True,
                    matched_mac="00:1B:44:11:3A:B7"
                )
                db.session.add(sample_log)

    db.session.commit()
    rebuild_vector_index()


# --------------------------------------------------------------------
# Authentication Routes
# --------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per minute")
def login():
    if current_user.is_authenticated:
        if current_user.is_teacher:
            return redirect(url_for("dashboard"))
        return redirect(url_for("student_portal"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        selected_role = request.form.get("role", "student").strip().lower()

        # Case-insensitive lookup for username/roll number
        user = User.query.filter(db.func.lower(User.username) == username.lower()).first()
        if user and user.check_password(password):
            # Strict role enforcement per tab
            if selected_role == "student" and user.is_teacher:
                flash("⛔ Access Denied: This is a Faculty/Admin account. Please switch to the 'Faculty / Admin' tab to sign in.", "warning")
                return render_template("login.html", active_role="student", prefill_username=username)

            if selected_role in ["faculty", "teacher", "admin"] and not user.is_teacher:
                flash("⛔ Access Denied: This is a Student account. Please switch to the 'Student Portal' tab to sign in.", "warning")
                return render_template("login.html", active_role="faculty", prefill_username=username)

            login_user(user)
            flash(f"Welcome back, {user.username} ({user.role.title()})!", "success")
            next_page = request.args.get("next")
            if next_page:
                return redirect(next_page)
            if user.is_teacher:
                return redirect(url_for("dashboard"))
            return redirect(url_for("student_portal"))
        else:
            flash("Invalid credentials. Please verify your Roll Number/Username and Password.", "danger")

    return render_template("login.html", active_role="student")




@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --------------------------------------------------------------------
# Core Pages
# --------------------------------------------------------------------

@app.route("/")
def home():
    if current_user.is_authenticated:
        if current_user.is_teacher:
            return redirect(url_for("dashboard"))
        return redirect(url_for("student_portal"))
    return redirect(url_for("login"))


# --------------------------------------------------------------------
# Multi-Embedding Student Registration
# --------------------------------------------------------------------

def generate_unique_roll_no():
    """Generates an academic roll number like '2026CS101', '2026CS102', etc."""
    year = datetime.now().year
    prefix = f"{year}CS"
    count = Student.query.count() + 1
    candidate = f"{prefix}{count:03d}"
    while Student.query.filter_by(roll_no=candidate).first():
        count += 1
        candidate = f"{prefix}{count:03d}"
    return candidate


@app.route("/register", methods=["GET", "POST"])
@teacher_required
def register():
    if request.method == "GET":
        next_roll = generate_unique_roll_no()
        return render_template("register.html", next_roll=next_roll)

    name = request.form.get("name", "").strip()
    roll_no = request.form.get("roll_no", "").strip()
    phone_number = request.form.get("phone_number", "").strip()
    birth_year = request.form.get("birth_year", "").strip()
    mac_address = request.form.get("mac_address", "").strip().upper()
    create_student_account = request.form.get("create_account", "true") == "true"

    if not roll_no:
        roll_no = generate_unique_roll_no()

    if not name:
        return jsonify({"ok": False, "error": "Full Name is required."}), 400

    if not phone_number or not birth_year:
        return jsonify({"ok": False, "error": "Phone number and Birth Year are required for student password derivation."}), 400

    # Auto password format: phone_number + birth_year (e.g. 98765432102003)
    # Clean non-digits for consistency
    clean_phone = re.sub(r"\D", "", phone_number)
    clean_year = re.sub(r"\D", "", birth_year)
    computed_password = f"{clean_phone}{clean_year}"

    # Supports 1 to 3 images from multi-angle capture
    images_raw = request.form.getlist("images[]")
    if not images_raw:
        single_img = request.form.get("image_data", "")
        if single_img:
            images_raw = [single_img]

    if not images_raw:
        return jsonify({"ok": False, "error": "No face photos provided. Capture all 3 angles."}), 400

    if not mac_address:
        mac_address = wifi_simulator.generate_mock_mac()
    elif not MAC_RE.match(mac_address):
        return jsonify({"ok": False, "error": "MAC address must look like AA:BB:CC:DD:EE:FF."}), 400

    if Student.query.filter_by(roll_no=roll_no).first():
        roll_no = generate_unique_roll_no()
    if Student.query.filter_by(mac_address=mac_address).first():
        return jsonify({"ok": False, "error": "That MAC address is already registered to another student."}), 400

    # Process all image angles
    try:
        decoded_images = [face_utils.decode_base64_image(img) for img in images_raw]
        embeddings = face_utils.get_multiple_embeddings(decoded_images)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Save primary photo (Center angle)
    photo_filename = f"{roll_no}.jpg"
    photo_path = os.path.join(PHOTO_DIR, photo_filename)
    from PIL import Image
    Image.fromarray(decoded_images[0]).save(photo_path)

    # Extract clean base64 data for persistent database storage
    primary_b64 = images_raw[0]
    if "," in primary_b64:
        primary_b64 = primary_b64.split(",", 1)[1]

    student = Student(
        name=name,
        roll_no=roll_no,
        phone_number=phone_number,
        birth_year=birth_year,
        mac_address=mac_address,
        photo_path=f"student_photos/{photo_filename}",
        photo_base64=primary_b64,
    )
    student.set_embeddings(embeddings)
    db.session.add(student)
    db.session.flush()

    # Add to FAISS Vector Index
    vector_index.add(embeddings, student.id)

    # Automatically create student login account with password = phone + birthyear
    if create_student_account:
        username = roll_no.lower()
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, role="student", student_id=student.id)
            user.set_password(computed_password)
            db.session.add(user)
        else:
            user.set_password(computed_password)

    db.session.commit()
    return jsonify({
        "ok": True,
        "student": student.to_dict(),
        "account_info": {
            "username": roll_no.lower(),
            "password": computed_password,
            "formula": "Phone Number + Birth Year"
        }
    })



@app.route("/api/wifi/connect", methods=["POST"])
@limiter.limit("30 per minute")
def wifi_connect():
    """Simulates student device connecting to classroom Wi-Fi with their MAC / Roll No."""
    mac_address = request.form.get("mac_address", "").strip().upper()
    roll_no = request.form.get("roll_no", "").strip()

    if roll_no and not mac_address:
        student = Student.query.filter_by(roll_no=roll_no).first()
        if student:
            mac_address = student.mac_address.upper()

    if not mac_address:
        return jsonify({"ok": False, "error": "MAC Address or Student Roll No required."}), 400

    wifi_simulator.register_connected_device(mac_address)
    return jsonify({
        "ok": True,
        "mac_address": mac_address,
        "message": f"Device ({mac_address}) successfully connected to Classroom AP."
    })


@app.route("/mark", methods=["GET", "POST"])
@limiter.limit("40 per minute")
def mark_attendance():
    if request.method == "GET":
        active_sessions = AttendanceSession.query.filter_by(is_active=True).all()
        students = Student.query.order_by(Student.name).all()
        return render_template("mark.html", active_sessions=active_sessions, students=[s.to_dict() for s in students])

    image_data = request.form.get("image_data", "")
    session_id = request.form.get("session_id", type=int)
    student_mac = request.form.get("mac_address", "").strip().upper()
    liveness_verified = request.form.get("liveness_verified", "false") == "true"

    if not image_data:
        return jsonify({"ok": False, "error": "No photo captured."}), 400

    if not liveness_verified:
        return jsonify({
            "ok": False,
            "error": "Liveness verification required. Please complete the quick challenge to confirm real person presence."
        }), 400

    # Validate Time Window if session specified
    target_session = None
    if session_id:
        target_session = AttendanceSession.query.get(session_id)
        if not target_session or not target_session.is_open():
            return jsonify({
                "ok": False,
                "error": f"Attendance window for '{target_session.title if target_session else session_id}' is currently CLOSED."
            }), 400

    threshold = float(SystemSetting.get_setting("match_threshold", "0.68"))

    try:
        image = face_utils.decode_base64_image(image_data)
        probe_embedding = face_utils.get_embedding(image)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Step 2: FAISS Vector Facial Recognition Match
    matched_student_id, score = vector_index.find_best_match(probe_embedding, threshold=threshold)

    if not matched_student_id:
        return jsonify({
            "ok": True,
            "recognized": False,
            "message": f"Face not recognized (similarity: {max(0.0, score):.2f}, required: {threshold:.2f}).",
            "score": round(max(0.0, score), 4),
        })

    matched_student = db.session.get(Student, matched_student_id)
    if not matched_student:
        return jsonify({"ok": False, "error": "Student record not found."}), 404

    # Step 1 Check: Verify if student's device is on the Classroom Wi-Fi Network
    students = Student.query.all()
    registered_macs = [s.mac_address for s in students]
    connected_macs = wifi_simulator.simulate_connected_macs(registered_macs)
    
    # Check if student's registered MAC or claimed MAC is verified on AP
    check_mac = student_mac if student_mac else matched_student.mac_address
    wifi_ok = wifi_simulator.verify_mac(check_mac, connected_macs)

    status = "PRESENT" if wifi_ok else "FLAGGED"

    log = AttendanceLog(
        student_id=matched_student.id,
        session_id=target_session.id if target_session else None,
        face_match_score=score,
        wifi_verified=wifi_ok,
        liveness_verified=liveness_verified,
        matched_mac=matched_student.mac_address if wifi_ok else None,
        status=status,
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        "ok": True,
        "recognized": True,
        "student": matched_student.to_dict(),
        "score": round(score, 4),
        "wifi_verified": wifi_ok,
        "liveness_verified": True,
        "session": target_session.title if target_session else "Ad-hoc Check-in",
        "status": status,
        "message": (
            f"✅ {matched_student.name} marked PRESENT (Wi-Fi Connected & Face Score {score:.2f} matched)."
            if wifi_ok else
            f"⚠️ {matched_student.name}'s face matched ({score:.2f}) but device was NOT connected to Classroom Wi-Fi — FLAGGED as possible proxy."
        ),
    })



# --------------------------------------------------------------------
# Session Time-Window Management (Teacher/Admin)
# --------------------------------------------------------------------

@app.route("/sessions", methods=["GET", "POST"])
@teacher_required
def manage_sessions():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        course_code = request.form.get("course_code", "").strip().upper()
        duration_minutes = int(request.form.get("duration", 15))

        if not title:
            flash("Session title is required.", "danger")
            return redirect(url_for("manage_sessions"))

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        session = AttendanceSession(
            title=title,
            course_code=course_code or "GEN101",
            start_time=start_time,
            end_time=end_time,
            is_active=True,
            created_by=current_user.username,
        )
        db.session.add(session)
        db.session.commit()
        flash(f"Active Attendance Window opened for '{title}' ({duration_minutes} mins).", "success")
        return redirect(url_for("manage_sessions"))

    sessions = AttendanceSession.query.order_by(AttendanceSession.start_time.desc()).all()
    return render_template("sessions.html", sessions=sessions)


@app.route("/sessions/<int:session_id>/toggle", methods=["POST"])
@teacher_required
def toggle_session(session_id):
    session = AttendanceSession.query.get_or_404(session_id)
    session.is_active = not session.is_active
    db.session.commit()
    status_str = "activated" if session.is_active else "closed"
    flash(f"Session '{session.title}' {status_str}.", "info")
    return redirect(url_for("manage_sessions"))


# --------------------------------------------------------------------
# System Threshold Settings (Teacher/Admin)
# --------------------------------------------------------------------

@app.route("/settings/threshold", methods=["POST"])
@teacher_required
def update_threshold():
    threshold = request.form.get("threshold", "0.68").strip()
    try:
        val = float(threshold)
        if 0.1 <= val <= 0.99:
            SystemSetting.set_setting("match_threshold", f"{val:.2f}")
            flash(f"Face match threshold updated to {val:.2f}.", "success")
        else:
            flash("Threshold must be between 0.10 and 0.99.", "danger")
    except ValueError:
        flash("Invalid threshold value.", "danger")
    return redirect(url_for("dashboard"))


# --------------------------------------------------------------------
# Admin Dashboard & Student Portal
# --------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    # If student, redirect to student portal
    if not current_user.is_teacher:
        return redirect(url_for("student_portal"))

    q = AttendanceLog.query
    student_id = request.args.get("student_id", type=int)
    day = request.args.get("date", "")
    status = request.args.get("status", "")
    session_id = request.args.get("session_id", type=int)

    if student_id:
        q = q.filter(AttendanceLog.student_id == student_id)
    if day:
        q = q.filter(db.func.date(AttendanceLog.timestamp) == day)
    if status:
        q = q.filter(AttendanceLog.status == status)
    if session_id:
        q = q.filter(AttendanceLog.session_id == session_id)

    logs = q.order_by(AttendanceLog.timestamp.desc()).limit(500).all()
    students = Student.query.order_by(Student.name).all()
    sessions = AttendanceSession.query.order_by(AttendanceSession.start_time.desc()).all()

    # Dashboard Metrics
    total_students = len(students)
    today_str = date.today().isoformat()
    today_present = AttendanceLog.query.filter(
        db.func.date(AttendanceLog.timestamp) == today_str,
        AttendanceLog.status == "PRESENT"
    ).count()
    today_flagged = AttendanceLog.query.filter(
        db.func.date(AttendanceLog.timestamp) == today_str,
        AttendanceLog.status == "FLAGGED"
    ).count()

    current_threshold = SystemSetting.get_setting("match_threshold", "0.68")

    return render_template(
        "dashboard.html",
        logs=[l.to_dict() for l in logs],
        students=students,
        sessions=sessions,
        metrics={
            "total_students": total_students,
            "today_present": today_present,
            "today_flagged": today_flagged,
            "attendance_rate": round((today_present / total_students * 100) if total_students > 0 else 0, 1),
        },
        current_threshold=current_threshold,
        filters={"student_id": student_id, "date": day, "status": status, "session_id": session_id},
    )


@app.route("/portal")
@login_required
def student_portal():
    student = current_user.student
    if not student and current_user.is_teacher:
        return redirect(url_for("dashboard"))

    if not student:
        flash("No student profile linked to this account.", "warning")
        return render_template("portal.html", logs=[], student=None, attendance_rate=0, active_sessions=[])

    logs = AttendanceLog.query.filter_by(student_id=student.id).order_by(AttendanceLog.timestamp.desc()).all()
    present_count = sum(1 for l in logs if l.status == "PRESENT")
    total_logs = len(logs)
    attendance_rate = round((present_count / total_logs * 100) if total_logs > 0 else 0, 1)

    active_sessions = AttendanceSession.query.filter_by(is_active=True).order_by(AttendanceSession.start_time.desc()).all()

    return render_template(
        "portal.html",
        student=student.to_dict(),
        logs=[l.to_dict() for l in logs],
        attendance_rate=attendance_rate,
        present_count=present_count,
        total_logs=total_logs,
        active_sessions=active_sessions,
    )


@app.route("/portal/checkin", methods=["POST"])
@login_required
@limiter.limit("40 per minute")
def portal_checkin():
    student = current_user.student
    if not student:
        return jsonify({"ok": False, "error": "No student profile linked to this account."}), 403

    image_data = request.form.get("image_data", "")
    session_id = request.form.get("session_id", type=int)
    liveness_verified = request.form.get("liveness_verified", "false") == "true"
    mac_address = (request.form.get("mac_address", "") or student.mac_address).strip().upper()

    if not image_data:
        return jsonify({"ok": False, "error": "No camera snapshot provided."}), 400

    if not liveness_verified:
        return jsonify({
            "ok": False,
            "error": "Liveness verification is required before submitting attendance."
        }), 400

    # Validate Time Window if session specified
    target_session = None
    if session_id:
        target_session = AttendanceSession.query.get(session_id)
        if not target_session or not target_session.is_open():
            return jsonify({
                "ok": False,
                "error": f"Attendance window for '{target_session.title if target_session else session_id}' is currently CLOSED."
            }), 400

    threshold = float(SystemSetting.get_setting("match_threshold", "0.68"))

    try:
        image = face_utils.decode_base64_image(image_data)
        probe_embedding = face_utils.get_embedding(image)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Match face in FAISS Vector Index
    matched_student_id, score = vector_index.find_best_match(probe_embedding, threshold=threshold)

    # Check if face matched any student
    if not matched_student_id:
        return jsonify({
            "ok": True,
            "recognized": False,
            "score": round(max(0.0, score), 4),
            "message": f"Face not recognized (Similarity: {max(0.0, score):.2f}, required: {threshold:.2f}). Please face the camera clearly in good lighting."
        })

    # Anti-Proxy Verification: Ensure face matches the LOGGED-IN student
    if matched_student_id != student.id:
        return jsonify({
            "ok": True,
            "recognized": False,
            "score": round(score, 4),
            "identity_mismatch": True,
            "message": f"⚠️ Identity Mismatch: Face detected does not match logged-in account for {student.name} ({student.roll_no}). Proxy attendance attempts are strictly prohibited."
        })

    # Step 1 Check: Verify if student's device is active on the Classroom Wi-Fi AP
    students = Student.query.all()
    registered_macs = [s.mac_address for s in students]
    connected_macs = wifi_simulator.simulate_connected_macs(registered_macs)
    
    wifi_ok = wifi_simulator.verify_mac(mac_address, connected_macs)
    status = "PRESENT" if wifi_ok else "FLAGGED"

    log = AttendanceLog(
        student_id=student.id,
        session_id=target_session.id if target_session else None,
        face_match_score=score,
        wifi_verified=wifi_ok,
        liveness_verified=liveness_verified,
        matched_mac=mac_address if wifi_ok else None,
        status=status,
    )
    db.session.add(log)
    db.session.commit()

    # Calculate updated attendance rate
    all_logs = AttendanceLog.query.filter_by(student_id=student.id).all()
    present_count = sum(1 for l in all_logs if l.status == "PRESENT")
    total_logs = len(all_logs)
    attendance_rate = round((present_count / total_logs * 100) if total_logs > 0 else 0, 1)

    return jsonify({
        "ok": True,
        "recognized": True,
        "student": student.to_dict(),
        "score": round(score, 4),
        "wifi_verified": wifi_ok,
        "liveness_verified": True,
        "session": target_session.title if target_session else "General Lecture",
        "status": status,
        "new_log": log.to_dict(),
        "attendance_rate": attendance_rate,
        "present_count": present_count,
        "total_logs": total_logs,
        "message": (
            f"✅ Attendance marked as PRESENT! Wi-Fi verified on campus AP and face matched with {score*100:.1f}% confidence."
            if wifi_ok else
            f"⚠️ Attendance marked as FLAGGED. Face matched ({score*100:.1f}%), but device ({mac_address}) was NOT connected to Classroom Wi-Fi."
        )
    })



@app.route("/students")
@teacher_required
def students_list():
    students = Student.query.order_by(Student.name).all()
    return render_template("students.html", students=[s.to_dict() for s in students])


@app.route("/students/<int:student_id>/delete", methods=["POST"])
@teacher_required
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    db.session.delete(student)
    db.session.commit()
    rebuild_vector_index()
    flash(f"Student {student.name} deleted.", "info")
    return redirect(url_for("students_list"))


@app.route("/export.csv")
@teacher_required
def export_csv():
    q = AttendanceLog.query
    student_id = request.args.get("student_id", type=int)
    day = request.args.get("date", "")
    status = request.args.get("status", "")

    if student_id:
        q = q.filter(AttendanceLog.student_id == student_id)
    if day:
        q = q.filter(db.func.date(AttendanceLog.timestamp) == day)
    if status:
        q = q.filter(AttendanceLog.status == status)

    logs = q.order_by(AttendanceLog.timestamp.desc()).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Timestamp", "Session", "Name", "Roll No", "Face Score", "Wi-Fi Verified", "Liveness Verified", "Status"])
    for l in logs:
        writer.writerow([
            l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            l.session.title if l.session else "Ad-hoc",
            l.student.name if l.student else "Unknown",
            l.student.roll_no if l.student else "-",
            f"{l.face_match_score:.4f}",
            "Yes" if l.wifi_verified else "No",
            "Yes" if l.liveness_verified else "No",
            l.status,
        ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendify_export.csv"},
    )


# --------------------------------------------------------------------
# 1. Manual Override & Academic Exception Auditing
# --------------------------------------------------------------------

@app.route("/attendance/override", methods=["POST"])
@teacher_required
@limiter.limit("30 per minute")
def manual_override():
    student_id = request.form.get("student_id", type=int)
    session_id = request.form.get("session_id", type=int)
    status = request.form.get("status", "PRESENT").strip().upper()
    reason = request.form.get("reason", "").strip() or f"Marked as {status}"

    if not student_id:
        return jsonify({"ok": False, "error": "Student ID is required."}), 400

    student = db.session.get(Student, student_id)
    if not student:
        return jsonify({"ok": False, "error": "Student not found."}), 404

    target_session = db.session.get(AttendanceSession, session_id) if session_id else None

    # Search for existing log for this student & session
    query = AttendanceLog.query.filter_by(student_id=student.id)
    if target_session:
        query = query.filter_by(session_id=target_session.id)
    else:
        # Check today's ad-hoc log
        query = query.filter(db.func.date(AttendanceLog.timestamp) == date.today().isoformat())

    existing_log = query.order_by(AttendanceLog.timestamp.desc()).first()

    if existing_log:
        existing_log.status = status
        existing_log.is_manual_override = True
        existing_log.override_by = current_user.username
        existing_log.override_reason = reason
        log = existing_log
    else:
        log = AttendanceLog(
            student_id=student.id,
            session_id=target_session.id if target_session else None,
            face_match_score=1.0,
            wifi_verified=True,
            liveness_verified=True,
            matched_mac=student.mac_address,
            status=status,
            is_manual_override=True,
            override_by=current_user.username,
            override_reason=reason,
        )
        db.session.add(log)

    db.session.commit()
    return jsonify({
        "ok": True,
        "message": f"Manual override applied: {student.name} ({student.roll_no}) status updated to '{status}' with audit trail.",
        "log": log.to_dict()
    })


# --------------------------------------------------------------------
# 2. Bulk Student Import (CSV / Excel)
# --------------------------------------------------------------------

@app.route("/students/import", methods=["GET", "POST"])
@teacher_required
def import_students():
    if request.method == "GET":
        return render_template("import_students.html")

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a valid CSV file to upload.", "danger")
        return redirect(url_for("import_students"))

    imported_count = 0
    skipped_count = 0
    errors = []

    try:
        content = file.stream.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        
        for row_idx, raw_row in enumerate(reader, start=2):
            row = {k.strip().lower().replace(" ", "_"): v.strip() for k, v in raw_row.items() if k}
            name = row.get("name") or row.get("full_name") or row.get("student_name")
            phone = row.get("phone") or row.get("phone_number") or row.get("mobile") or ""
            birth_year = row.get("birth_year") or row.get("year") or row.get("dob_year") or ""
            roll_no = row.get("roll_no") or row.get("roll_number") or row.get("roll")
            mac_address = row.get("mac_address") or row.get("mac")

            if not name:
                errors.append(f"Row {row_idx}: Skipped (Student Name missing).")
                skipped_count += 1
                continue

            if not roll_no:
                errors.append(f"Row {row_idx}: Skipped (Roll Number is required).")
                skipped_count += 1
                continue

            if not mac_address:
                errors.append(f"Row {row_idx}: Skipped (MAC Address is required).")
                skipped_count += 1
                continue

            roll_no = roll_no.strip().upper()
            mac_address = mac_address.strip().upper()

            if not MAC_RE.match(mac_address):
                errors.append(f"Row {row_idx}: Skipped (Invalid MAC '{mac_address}', must look like AA:BB:CC:DD:EE:FF).")
                skipped_count += 1
                continue

            # If roll_no or MAC already exists, skip
            if Student.query.filter_by(roll_no=roll_no).first():
                errors.append(f"Row {row_idx}: Skipped (Roll No '{roll_no}' already exists).")
                skipped_count += 1
                continue

            if Student.query.filter_by(mac_address=mac_address).first():
                errors.append(f"Row {row_idx}: Skipped (MAC Address '{mac_address}' already registered).")
                skipped_count += 1
                continue

            phone = phone or "9876543210"
            birth_year = birth_year or "2005"
            clean_phone = re.sub(r"\D", "", phone)
            clean_year = re.sub(r"\D", "", birth_year)
            derived_password = f"{clean_phone}{clean_year}"

            student = Student(
                name=name,
                roll_no=roll_no,
                phone_number=phone,
                birth_year=birth_year,
                mac_address=mac_address,
                photo_path="student_photos/default.png",
                enrollment_status="PENDING_BIOMETRICS",
            )
            student.set_embeddings([])
            db.session.add(student)
            db.session.flush()

            # Auto-provision student user account
            username = roll_no.lower()
            user = User.query.filter_by(username=username).first()
            if not user:
                user = User(username=username, role="student", student_id=student.id)
                user.set_password(derived_password)
                db.session.add(user)

            imported_count += 1

        db.session.commit()
        flash(f"Successfully imported {imported_count} student accounts ({skipped_count} skipped).", "success")
        return render_template("import_students.html", imported_count=imported_count, errors=errors)

    except Exception as e:
        db.session.rollback()
        flash(f"CSV Import Error: {str(e)}", "danger")
        return redirect(url_for("import_students"))


@app.route("/students/import/template")
@teacher_required
def download_import_template():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Roll_No", "Phone_Number", "Birth_Year", "MAC_Address"])
    writer.writerow(["Aarav Sharma", "2026CS101", "9876543210", "2005", "AA:BB:CC:11:22:33"])
    writer.writerow(["Diya Patel", "2026CS102", "9876543211", "2004", "11:22:33:44:55:66"])
    writer.writerow(["Rohan Gupta", "2026CS103", "9876543212", "2005", "DD:EE:FF:44:55:66"])
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendify_students_template.csv"}
    )


# --------------------------------------------------------------------
# 3. Course & Subject Timetable Scheduling
# --------------------------------------------------------------------

@app.route("/timetable")
@teacher_required
def timetable():
    courses = Course.query.order_by(Course.code).all()
    slots = TimetableSlot.query.all()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    # Organize slots by day
    schedule_by_day = {d: [] for d in days}
    for s in slots:
        if s.day_of_week in schedule_by_day:
            schedule_by_day[s.day_of_week].append(s)
            
    return render_template(
        "timetable.html",
        courses=courses,
        slots=slots,
        days=days,
        schedule_by_day=schedule_by_day
    )


@app.route("/courses/create", methods=["POST"])
@teacher_required
def create_course():
    code = request.form.get("code", "").strip().upper()
    title = request.form.get("title", "").strip()
    department = request.form.get("department", "Computer Science & Engineering").strip()
    instructor = request.form.get("instructor", current_user.username).strip()

    if not code or not title:
        flash("Course Code and Title are required.", "danger")
        return redirect(url_for("timetable"))

    if Course.query.filter_by(code=code).first():
        flash(f"Course '{code}' already exists.", "warning")
        return redirect(url_for("timetable"))

    course = Course(code=code, title=title, department=department, instructor=instructor)
    db.session.add(course)
    db.session.commit()
    flash(f"Academic Course '{code}: {title}' created.", "success")
    return redirect(url_for("timetable"))


@app.route("/courses/<int:course_id>/delete", methods=["POST"])
@teacher_required
def delete_course(course_id):
    course = db.session.get(Course, course_id)
    if course:
        db.session.delete(course)
        db.session.commit()
        flash(f"Course '{course.code}' removed.", "info")
    return redirect(url_for("timetable"))


@app.route("/timetable/slot/create", methods=["POST"])
@teacher_required
def create_timetable_slot():
    course_id = request.form.get("course_id", type=int)
    day_of_week = request.form.get("day_of_week", "Monday")
    start_time = request.form.get("start_time", "10:00").strip()
    end_time = request.form.get("end_time", "11:00").strip()
    room = request.form.get("room", "Lecture Hall 101").strip()

    if not course_id or not start_time or not end_time:
        flash("Course, Day, and Time are required.", "danger")
        return redirect(url_for("timetable"))

    slot = TimetableSlot(
        course_id=course_id,
        day_of_week=day_of_week,
        start_time=start_time,
        end_time=end_time,
        room=room
    )
    db.session.add(slot)
    db.session.commit()
    flash("Weekly lecture timetable slot added.", "success")
    return redirect(url_for("timetable"))


@app.route("/timetable/slot/<int:slot_id>/delete", methods=["POST"])
@teacher_required
def delete_timetable_slot(slot_id):
    slot = db.session.get(TimetableSlot, slot_id)
    if slot:
        db.session.delete(slot)
        db.session.commit()
        flash("Timetable slot removed.", "info")
    return redirect(url_for("timetable"))


@app.route("/timetable/slot/<int:slot_id>/launch", methods=["POST"])
@teacher_required
def launch_slot_session(slot_id):
    slot = db.session.get(TimetableSlot, slot_id)
    if not slot or not slot.course:
        flash("Invalid timetable slot.", "danger")
        return redirect(url_for("timetable"))

    duration_mins = int(request.form.get("duration", 60))
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(minutes=duration_mins)

    session = AttendanceSession(
        title=f"{slot.course.code} — {slot.course.title} ({slot.room})",
        course_code=slot.course.code,
        course_id=slot.course.id,
        start_time=start_time,
        end_time=end_time,
        is_active=True,
        created_by=current_user.username
    )
    db.session.add(session)
    db.session.commit()
    flash(f"🔴 Live Attendance Window opened for '{slot.course.code}' ({duration_mins} mins) in {slot.room}!", "success")
    return redirect(url_for("manage_sessions"))


# --------------------------------------------------------------------
# 4. Official University PDF & Printable Report Generator
# --------------------------------------------------------------------

@app.route("/reports/official")
@teacher_required
def official_report():
    course_id = request.args.get("course_id", type=int)
    courses = Course.query.order_by(Course.code).all()
    students = Student.query.order_by(Student.roll_no).all()

    session_q = AttendanceSession.query
    if course_id:
        session_q = session_q.filter_by(course_id=course_id)
    sessions = session_q.order_by(AttendanceSession.start_time.desc()).all()
    total_lectures = len(sessions) if sessions else 1

    report_data = []
    defaulters_count = 0

    for s in students:
        log_q = AttendanceLog.query.filter_by(student_id=s.id)
        if course_id:
            log_q = log_q.join(AttendanceSession).filter(AttendanceSession.course_id == course_id)
        
        all_logs = log_q.all()
        present_count = sum(1 for l in all_logs if l.status == "PRESENT")
        overrides_count = sum(1 for l in all_logs if l.is_manual_override)
        
        rate = round((present_count / total_lectures * 100) if total_lectures > 0 else 0, 1)
        is_defaulter = rate < 75.0
        if is_defaulter:
            defaulters_count += 1

        report_data.append({
            "student": s,
            "present_count": present_count,
            "total_lectures": total_lectures,
            "overrides_count": overrides_count,
            "attendance_rate": rate,
            "is_defaulter": is_defaulter,
            "status": "ELIGIBLE" if not is_defaulter else "DEFAULTER (<75%)"
        })

    selected_course = db.session.get(Course, course_id) if course_id else None

    return render_template(
        "official_report.html",
        courses=courses,
        selected_course=selected_course,
        report_data=report_data,
        total_students=len(students),
        total_lectures=total_lectures,
        defaulters_count=defaulters_count,
        generated_at=datetime.now().strftime("%B %d, %Y - %H:%M"),
        dean_name="Prof. Alan Turing, Dean of Academic Affairs"
    )


@app.route("/reports/download-pdf")
@teacher_required
def download_pdf_report():
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    course_id = request.args.get("course_id", type=int)
    students = Student.query.order_by(Student.roll_no).all()
    selected_course = db.session.get(Course, course_id) if course_id else None

    session_q = AttendanceSession.query
    if course_id:
        session_q = session_q.filter_by(course_id=course_id)
    total_lectures = session_q.count() or 1

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()

    # Title & Header
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=16, alignment=1, textColor=colors.HexColor('#0b1329'))
    sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=10, alignment=1, textColor=colors.HexColor('#475569'))
    
    story.append(Paragraph("<b>UNIVERSITAS ATTENDIFY — OFFICIAL ACADEMIC ATTENDANCE REGISTER</b>", title_style))
    course_name = f"{selected_course.code}: {selected_course.title}" if selected_course else "All Registered Courses"
    story.append(Paragraph(f"Course: <b>{course_name}</b> | Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b> | Total Sessions Conducted: <b>{total_lectures}</b>", sub_style))
    story.append(Spacer(1, 15))

    # Table Header & Rows
    table_data = [["Roll No", "Student Name", "MAC Address", "Attended", "Total", "Rate (%)", "Audit Status"]]
    for s in students:
        log_q = AttendanceLog.query.filter_by(student_id=s.id)
        if course_id:
            log_q = log_q.join(AttendanceSession).filter(AttendanceSession.course_id == course_id)
        present = sum(1 for l in log_q.all() if l.status == "PRESENT")
        rate = round((present / total_lectures * 100), 1)
        status = "ELIGIBLE" if rate >= 75.0 else "DEFAULTER (<75%)"
        table_data.append([
            s.roll_no,
            s.name,
            s.mac_address,
            str(present),
            str(total_lectures),
            f"{rate}%",
            status
        ])

    t = Table(table_data, colWidths=[90, 160, 130, 70, 60, 80, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0b1329')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t)
    doc.build(story)

    buffer.seek(0)
    filename = f"Official_Attendance_Register_{selected_course.code if selected_course else 'All'}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# --------------------------------------------------------------------
# Static Assets & Entrypoint
# --------------------------------------------------------------------

@app.route("/data/student_photos/<path:filename>")
def student_photo(filename):
    file_path = os.path.join(PHOTO_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(PHOTO_DIR, filename)
    
    # Attempt to load from database if file is missing (e.g. after container restart)
    student = Student.query.filter(Student.photo_path.like(f"%{filename}")).first()
    if student and student.photo_base64:
        try:
            raw_data = base64.b64decode(student.photo_base64)
            return Response(raw_data, mimetype="image/jpeg")
        except Exception:
            pass

    # Clean default SVG avatar fallback
    svg_avatar = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
        <circle cx="50" cy="50" r="50" fill="#4f46e5"/>
        <circle cx="50" cy="38" r="18" fill="#ffffff"/>
        <path d="M20 85 C20 62, 80 62, 80 85 Z" fill="#ffffff"/>
    </svg>'''
    return Response(svg_avatar, mimetype="image/svg+xml")


@app.route("/favicon.ico")
def favicon():
    svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <text y="0.9em" font-size="90">📸</text>
    </svg>'''
    return Response(svg_icon, mimetype="image/svg+xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)


