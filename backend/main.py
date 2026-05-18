from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
from dotenv import load_dotenv
import bcrypt
import json, os

from typing import Optional, List

from database import SessionLocal, engine
import models
from predict import predict_stages, STAGES

load_dotenv()
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FAILSAFE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "failsafe-dev-secret")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 8

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def hash_password(password: str) -> str:
    # bcrypt has a 72-byte limit; truncate to be safe
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    pw = password.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except Exception:
        return False


# ── helpers ──────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── auth ─────────────────────────────────────────────────────────────────────

class RegisterReq(BaseModel):
    name: str
    email: str
    password: str
    role: str = "faculty"


@app.post("/auth/register")
def register(req: RegisterReq, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = models.User(
        name=req.name,
        email=req.email,
        password_hash=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Registered", "id": user.id}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    token = create_token(user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "name": user.name,
        "role": user.role,
    }


@app.get("/auth/me")
def me(current_user: models.User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role,
    }


# ── prediction ────────────────────────────────────────────────────────────────

class PredictReq(BaseModel):
    student_name: str
    school:     str
    sex:        str
    age:        int
    address:    str
    famsize:    str
    Pstatus:    str
    Medu:       int
    Fedu:       int
    Mjob:       str
    Fjob:       str
    reason:     str
    guardian:   str
    traveltime: int
    studytime:  int
    failures:   int
    schoolsup:  str
    famsup:     str
    paid:       str
    activities: str
    nursery:    str
    higher:     str
    internet:   str
    romantic:   str
    famrel:     int
    freetime:   int
    goout:      int
    Dalc:       int
    Walc:       int
    health:     int
    absences:   int
    subject:    str
    G1:         Optional[int] = None
    G2:         Optional[int] = None
    stages:     Optional[List[str]] = None  # subset of {pre, mid, full}; default = auto


@app.post("/predict")
def predict(
    req: PredictReq,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    raw = req.model_dump()
    student_name = raw.pop("student_name")
    stages_req   = raw.pop("stages", None)

    if stages_req:
        bad = [s for s in stages_req if s not in STAGES]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown stage(s): {bad}")

    results = predict_stages(raw, stages_req)
    if not results:
        raise HTTPException(status_code=400, detail="No model can run on the supplied inputs.")

    # Persist the latest (highest-information) stage as the canonical row.
    canonical = results[-1]

    pred = models.Prediction(
        user_id=current_user.id,
        student_name=student_name,
        subject=raw["subject"],
        features=json.dumps(raw, default=str),
        risk_prob=canonical["risk_prob"],
        risk_level=canonical["risk_level"],
        interventions=json.dumps(canonical["interventions"]),
        shap_top5=json.dumps(canonical["shap_top5"], default=str),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return {
        "id":           pred.id,
        "student_name": student_name,
        "results":      results,        # one entry per stage that ran
        "canonical":    canonical,      # convenience: the latest-stage result
    }


@app.get("/predictions")
def get_predictions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    rows = (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == current_user.id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )
    return [
        {
            "id": p.id,
            "student_name": p.student_name,
            "subject": p.subject,
            "risk_prob": p.risk_prob,
            "risk_level": p.risk_level,
            "interventions": json.loads(p.interventions),
            "shap_top5": json.loads(p.shap_top5),
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in rows
    ]


@app.get("/dashboard/stats")
def stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    rows = db.query(models.Prediction).filter(models.Prediction.user_id == current_user.id).all()
    return {
        "total":  len(rows),
        "high":   sum(1 for p in rows if p.risk_level == "High"),
        "medium": sum(1 for p in rows if p.risk_level == "Medium"),
        "low":    sum(1 for p in rows if p.risk_level == "Low"),
    }


@app.delete("/predictions/{pred_id}")
def delete_prediction(
    pred_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    pred = db.query(models.Prediction).filter(
        models.Prediction.id == pred_id,
        models.Prediction.user_id == current_user.id,
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(pred)
    db.commit()
    return {"message": "Deleted"}
