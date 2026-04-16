import time
import cv2
import streamlit as st
from chatbot import ask_fitbot, reset_chat
from main import process_poseguard_frame

st.set_page_config(page_title="PoseGuard Dashboard", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
div[data-testid="stMetric"] {
    background-color: #111827;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "camera_started_once" not in st.session_state:
    st.session_state.camera_started_once = False


def start_detection():
    st.session_state.run_detection = True
    st.session_state.camera_started_once = True


def stop_detection():
    st.session_state.run_detection = False


st.sidebar.title("PoseGuard")
page = st.sidebar.radio("Navigate", ["Live Demo", "FitBot", "About"])
demo_mode = st.sidebar.toggle("Demo Mode (no camera)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("**How to use**")
st.sidebar.write("1. Open Live Demo")
st.sidebar.write("2. Click Start Detection")
st.sidebar.write("3. Do your exercise in front of laptop camera")
st.sidebar.write("4. Open this same dashboard on phone if needed")

st.title("🏋️ PoseGuard")
st.caption("AI Gym Posture Detection • Live Feedback • FitBot")

if page == "Live Demo":
    left, right = st.columns([1.45, 1])

    with left:
        st.header("📹 Live Detection")

        b1, b2 = st.columns(2)
        with b1:
            st.button("Start Detection", on_click=start_detection, use_container_width=True)
        with b2:
            st.button("Stop Detection", on_click=stop_detection, use_container_width=True)

        frame_placeholder = st.empty()
        metric_placeholder = st.empty()
        note_placeholder = st.empty()

    with right:
        st.header("🤖 FitBot")

        user_input = st.text_input("Ask a fitness question")

        c1, c2 = st.columns(2)
        with c1:
            ask_btn = st.button("Ask FitBot", use_container_width=True)
        with c2:
            reset_btn = st.button("Reset Chat", use_container_width=True)

        if ask_btn:
            if user_input.strip():
                reply = ask_fitbot(user_input)
                st.session_state.chat_messages.append(("You", user_input))
                st.session_state.chat_messages.append(("FitBot", reply))
            else:
                st.warning("Please enter a question.")

        if reset_btn:
            reset_chat()
            st.session_state.chat_messages = []

        if st.session_state.chat_messages:
            st.markdown("### Chat")
            for sender, message in st.session_state.chat_messages:
                st.write(f"**{sender}:** {message}")
        else:
            st.info("Ask FitBot about workouts, posture, warm-up, or nutrition.")

    st.markdown("---")
    st.subheader("📊 Project Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Project", "PoseGuard")
    s2.metric("Exercises", "Squat / Curl / Lunge / Bicep_curl")
    s3.metric("Assistant", "FitBot Active")

    if demo_mode:
        t = int(time.time()) % 20
        exercise = "SQUAT"
        reps = t // 4
        status = "CORRECT" if t % 2 == 0 else "WRONG"

        demo_frame = 255 * (cv2.imread("non_existing_file.png") if False else None)
        # fallback clean placeholder using Streamlit text block instead of image
        frame_placeholder.info("Demo Mode is ON. Camera is disabled. Showing simulated exercise data.")

        with metric_placeholder.container():
            x1, x2, x3 = st.columns(3)
            x1.metric("Exercise", exercise)
            x2.metric("Reps", reps)
            x3.metric("Status", status)

        note_placeholder.warning("Demo Mode is active. Use this if webcam is unavailable during presentation.")

    elif st.session_state.run_detection:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            frame_placeholder.error("Could not open webcam.")
            st.session_state.run_detection = False
        else:
            note_placeholder.success("Detection is running from your laptop webcam.")

            stop_button_holder = st.empty()

            while st.session_state.run_detection:
                ret, frame = cap.read()
                if not ret:
                    frame_placeholder.error("Could not read webcam frame.")
                    break

                processed_frame, exercise, reps, status = process_poseguard_frame(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                with metric_placeholder.container():
                    x1, x2, x3 = st.columns(3)
                    x1.metric("Exercise", exercise)
                    x2.metric("Reps", reps)
                    x3.metric("Status", status)

                with stop_button_holder.container():
                    if st.button("Stop Live Camera", key="stop_live_camera", use_container_width=True):
                        st.session_state.run_detection = False
                        break

            cap.release()
            cv2.destroyAllWindows()

    else:
        frame_placeholder.info("Click Start Detection to begin live posture analysis.")
        with metric_placeholder.container():
            x1, x2, x3 = st.columns(3)
            x1.metric("Exercise", "-")
            x2.metric("Reps", "0")
            x3.metric("Status", "WAITING")

        if st.session_state.camera_started_once:
            note_placeholder.warning("Detection stopped.")
        else:
            note_placeholder.caption("Tip: Use your laptop webcam for live analysis and open this dashboard on your phone for demo viewing.")

elif page == "FitBot":
    st.header("🤖 FitBot Assistant")
    st.write("Ask fitness questions about posture, sets, reps, form, workouts, or nutrition.")

    user_input = st.text_input("Ask FitBot anything", key="fitbot_page_input")

    c1, c2 = st.columns(2)
    with c1:
        ask_btn = st.button("Ask FitBot", key="fitbot_page_ask", use_container_width=True)
    with c2:
        reset_btn = st.button("Reset Chat", key="fitbot_page_reset", use_container_width=True)

    if ask_btn:
        if user_input.strip():
            reply = ask_fitbot(user_input)
            st.session_state.chat_messages.append(("You", user_input))
            st.session_state.chat_messages.append(("FitBot", reply))
        else:
            st.warning("Please enter a question.")

    if reset_btn:
        reset_chat()
        st.session_state.chat_messages = []

    if st.session_state.chat_messages:
        for sender, message in st.session_state.chat_messages:
            st.write(f"**{sender}:** {message}")
    else:
        st.info("Example: How do I improve squat form?")

elif page == "About":
    st.header("📘 About PoseGuard")
    st.write("""
PoseGuard is an AI-powered fitness assistant designed to help users improve gym posture and exercise technique.

### Key Features
- Real-time posture detection using computer vision
- Rep counting for selected exercises
- Live posture feedback
- AI chatbot for workout guidance
- Mobile browser demo through local network

### Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Streamlit
- Groq API

### Demo Flow
- Run the dashboard on your laptop
- Start live detection using laptop webcam
- Open the same dashboard on your phone using your laptop IP
- Show chatbot + posture metrics in real time
""")