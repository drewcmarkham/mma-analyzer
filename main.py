import cv2
import numpy as np
import math
import mediapipe as mp

# --------- Helpers ---------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# MediaPipe landmark indices (Pose)
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose.py
LMS = mp_pose.PoseLandmark

def to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def angle_deg(a, b, c):
    # angle at b (a-b-c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def safe_get(landmarks, idx, w, h):
    lm = landmarks[idx.value]
    return to_np(lm, w, h), lm.visibility

def put_text(img, text, x, y):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

# --------- Simple “vibe” rules ---------
def evaluate_form(landmarks, w, h):
    """
    Returns a list of (message, anchor_point) suggestions.
    All thresholds are vibe-y: tune them by eye after you see your own camera framing.
    """
    msgs = []

    # Get key points
    L_ank, v1 = safe_get(landmarks, LMS.LEFT_ANKLE, w, h)
    R_ank, v2 = safe_get(landmarks, LMS.RIGHT_ANKLE, w, h)
    L_sho, _  = safe_get(landmarks, LMS.LEFT_SHOULDER, w, h)
    R_sho, _  = safe_get(landmarks, LMS.RIGHT_SHOULDER, w, h)
    L_wri, _  = safe_get(landmarks, LMS.LEFT_WRIST, w, h)
    R_wri, _  = safe_get(landmarks, LMS.RIGHT_WRIST, w, h)
    L_elb, _  = safe_get(landmarks, LMS.LEFT_ELBOW, w, h)
    R_elb, _  = safe_get(landmarks, LMS.RIGHT_ELBOW, w, h)
    NOSE, _   = safe_get(landmarks, LMS.NOSE, w, h)
    L_hip, _  = safe_get(landmarks, LMS.LEFT_HIP, w, h)
    R_hip, _  = safe_get(landmarks, LMS.RIGHT_HIP, w, h)

    shoulder_width = dist(L_sho, R_sho) + 1e-6
    hip_width = dist(L_hip, R_hip) + 1e-6
    stance_width = dist(L_ank, R_ank)

    # 1) Stance width (basic fight stance ≈ ~1.2–1.8× shoulder/hip width depending on frame)
    # Using hip width because shoulder can shrink with camera angle
    if stance_width < 1.1 * hip_width:
        msgs.append(("Widen stance", ( (L_ank[0] + R_ank[0]) / 2, (L_ank[1] + R_ank[1]) / 2 )))
    elif stance_width > 2.2 * hip_width:
        msgs.append(("Narrow stance", ( (L_ank[0] + R_ank[0]) / 2, (L_ank[1] + R_ank[1]) / 2 )))

    # 2) Guard height (hands at or above chin/nose level)
    chin_y = NOSE[1] + 0.06 * h  # nose + small offset ≈ chin-ish
    if L_wri[1] > chin_y:
        msgs.append(("Left hand up!", (L_wri[0], L_wri[1])))
    if R_wri[1] > chin_y:
        msgs.append(("Right hand up!", (R_wri[0], R_wri[1])))

    # 3) Elbow structure on “jab” (check lead arm straightness when hand is forward of shoulder)
    # We can treat whichever wrist is more forward (smaller x if camera is mirrored) as the "jab" side.
    # Because frames are flipped for selfie view, we compare absolute forward by horizontal distance from mid-shoulder line.
    mid_shoulder_x = (L_sho[0] + R_sho[0]) / 2
    # Consider "punching" if wrist is ahead of its shoulder by a fraction of shoulder width
    def jab_check(shoulder, elbow, wrist, label):
        ahead = abs(wrist[0] - mid_shoulder_x) > 0.4 * shoulder_width
        if ahead:
            ang = angle_deg(shoulder, elbow, wrist)  # ~180 is straight
            if ang < 150:  # bent too much when extended
                msgs.append((f"Extend {label} jab", ((elbow[0]+wrist[0])/2, (elbow[1]+wrist[1])/2)))

    jab_check(L_sho, L_elb, L_wri, "left")
    jab_check(R_sho, R_elb, R_wri, "right")

    # 4) Hands returning to guard quickly after extension (simple: if wrist far ahead AND below chin → call out)
    def return_to_guard(wrist, label):
        if abs(wrist[0] - mid_shoulder_x) > 0.55 * shoulder_width and wrist[1] > chin_y:
            msgs.append((f"Return {label} to guard", (wrist[0], wrist[1])))

    return_to_guard(L_wri, "left")
    return_to_guard(R_wri, "right")

    # 5) Upright posture hint (hips roughly under shoulders vertically)
    mid_shoulder_y = (L_sho[1] + R_sho[1]) / 2
    mid_hip_y = (L_hip[1] + R_hip[1]) / 2
    # if hips are way behind shoulders (camera dependent), just give a gentle cue
    if mid_hip_y < mid_shoulder_y - 0.03 * h:
        msgs.append(("Lower center of gravity", ((L_hip[0]+R_hip[0])/2, (mid_hip_y))))

    return msgs

# --------- Main loop ---------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Higher res helps landmark stability; you can lower if laggy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(
        model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for "selfie" feel
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            if res.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

                # Evaluate rules
                msgs = evaluate_form(res.pose_landmarks.landmark, w, h)

                # Render messages
                for text, (x, y) in msgs:
                    put_text(frame, text, x + 10, y - 10)

            # UI text
            cv2.putText(frame, "AI MMA Coach - press 'q' to quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("MMA AI Coach", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
