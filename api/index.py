import os
import json
import base64
import tempfile
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from groq import Groq
import cv2

# Load local environment variables if present
load_dotenv()

# Point to the root-level templates directory relative to the serverless api function
app = Flask(__name__, template_folder='../templates')

# Cap upload size for the video analysis endpoint (200 MB) so a stray large file
# doesn't hang the request pipeline or exhaust server memory.
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Initialize API Keys securely from the production environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# --- Video Analysis Configuration ---
# Groq's multimodal vision model. Limited to a maximum of 5 images per request,
# so frames are sampled from the uploaded video and analyzed in small batches.
VISION_MODEL = "qwen/qwen3.6-27b"
VISION_BATCH_SIZE = 5
MAX_VIDEO_FRAMES = 15
MAX_FRAME_WIDTH = 800
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.webm', '.mkv'}

@app.route('/')
def home():
    """Renders the main single-page sports coach dashboard."""
    return render_template(
        'index.html', 
        supabase_url=SUPABASE_URL, 
        supabase_anon_key=SUPABASE_ANON_KEY
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Handshake validation endpoint for the frontend client."""
    return jsonify({
        "status": "healthy",
        "environment": "US-Sports-Ecosystem",
        "framework": "Flask Serverless",
        "api_configured": bool(GROQ_API_KEY)
    })

@app.route('/api/debrief', methods=['POST'])
def match_debrief():
    """Processes match narratives and returns structured tactical insights via Groq."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        narrative = data.get("narrative", "").strip()
        error_type = data.get("error_type", "Unforced errors")
        opponent_type = data.get("opponent_type", "Baseliner")
        player_level = data.get("player_level", "High School Varsity / NTRP 3.5")

        if not narrative:
            return jsonify({"error": "Match narrative input payload cannot be blank."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an expert US Tennis Solutions Architect and Strategic Performance Coach specializing "
            "in the US High School (Varsity/JV) and NCAA competitive ecosystem. Analyze the player's match performance "
            "narrative, main error classification, and opponent playstyle based on standard NTRP guidelines. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown blocks outside it:\n"
            "{\n"
            "  \"readiness_score\": 78,\n"
            "  \"motivational_tip\": \"Keep your feet active through the baseline transitions. Next set is yours!\",\n"
            "  \"weaknesses\": [\"Explanation of tactical breakdown 1\", \"Explanation of tactical breakdown 2\"],\n"
            "  \"tactics\": [\"Actionable adjustment 1\", \"Actionable adjustment 2\", \"Actionable adjustment 3\"]\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Opponent Archetype: {opponent_type}\n"
            f"Primary Error Focus: {error_type}\n"
            f"Match Narrative: {narrative}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Analysis pipeline failure: {str(e)}"}), 500

@app.route('/api/training', methods=['POST'])
def generate_training():
    """Generates a structured US tennis drill, fitness schedule, and standard high-performance nutritional plan."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        focus_area = data.get("focus_area", "Serves")
        time_available = data.get("time_available", "30 mins a day")
        gym_access = data.get("gym_access", "Full gym access")
        player_level = data.get("player_level", "High School Varsity / NTRP 3.5")
        # Optional personalization context sourced from the athlete's saved profile.
        # These are additive and default to empty strings so existing callers of this
        # endpoint that don't send them keep working exactly as before.
        player_weaknesses = data.get("weaknesses", "").strip()
        recent_injuries = data.get("recent_injuries", "").strip()
        favorite_serve = data.get("favorite_serve", "").strip()
        coach_notes = data.get("coach_notes", "").strip()

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an elite US tennis performance coordinator and high-performance sports nutritionist. "
            "Create a target development schedule using standard US metrics, common athletic fitness terms, and widely available US nutritional staples. "
            "When personalization context (known weaknesses, recent injuries, favorite serve, or coach notes) is provided, "
            "tailor the drills and exercises specifically to address those points rather than giving generic suggestions. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown formatting:\n"
            "{\n"
            "  \"drills\": [\"Drill 1 Description\", \"Drill 2 Description\"],\n"
            "  \"gym_exercises\": [\"Exercise 1 Description\", \"Exercise 2 Description\"],\n"
            "  \"pre_match_meals\": [\"Pre-Match Meal Combo 1\", \"Pre-Match Meal Combo 2\"],\n"
            "  \"post_match_meals\": [\"Post-Match Recovery 1\", \"Post-Match Recovery 2\"]\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Target Technical Focus: {focus_area}\n"
            f"Daily Time Constraints: {time_available}\n"
            f"Physical Gym Infrastructure Setup: {gym_access}\n"
            f"Known Weaknesses (personalize around these): {player_weaknesses or 'Not specified'}\n"
            f"Recent Injuries (respect these constraints): {recent_injuries or 'None reported'}\n"
            f"Favorite Serve: {favorite_serve or 'Not specified'}\n"
            f"Coach Notes: {coach_notes or 'None provided'}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Training hub pipeline failure: {str(e)}"}), 500

@app.route('/api/recovery', methods=['POST'])
def injury_recovery():
    """Generates a structured day-to-day injury recovery timeline and advice via Groq."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        injury = data.get("injury", "").strip()
        context = data.get("context", "").strip()
        player_level = data.get("player_level", "Ordinary / Recreational")

        if not injury:
            return jsonify({"error": "Injury type or classification name cannot be blank."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        system_instruction = (
            "You are an elite US sports physical therapist, athletic trainer, and tennis injury rehabilitation specialist. "
            "Analyze the player's tennis-related pain or injury context and produce a structured, safe, progressive recovery schedule. "
            "Keep the suggestions clinical, practical, and highly focused on tennis bio-mechanics. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown formatting:\n"
            "{\n"
            "  \"injury_name\": \"Injury Title / Area\",\n"
            "  \"recovery_plan\": [\n"
            "    {\n"
            "      \"day\": \"Phase or Day Range (e.g., Day 1-2: Acute Protection)\",\n"
            "      \"focus\": \"Short, clinical clinical target (e.g., Swelling Control & Tissue Calm)\",\n"
            "      \"protocols\": [\"Protocol action item 1\", \"Protocol action item 2\", \"Protocol action item 3\"]\n"
            "    }\n"
            "  ],\n"
            "  \"disclaimer\": \"Informational therapeutic outline only. Always seek counsel from an athletic medical professional before returning to court.\"\n"
            "}"
        )

        user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Reported Injury/Pain: {injury}\n"
            f"Symptoms & Context Details: {context}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Injury recovery pipeline failure: {str(e)}"}), 500

@app.route('/api/profile-insight', methods=['POST'])
def profile_insight():
    """Generates a personalized welcome-back readiness briefing from the player's saved profile and match history."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        full_name = (data.get("full_name") or "Athlete").strip() or "Athlete"
        height = (data.get("height") or "").strip()
        play_style = (data.get("play_style") or "").strip()
        favorite_serve = (data.get("favorite_serve") or "").strip()
        weaknesses = (data.get("weaknesses") or "").strip()
        recent_injuries = (data.get("recent_injuries") or "").strip()
        coach_notes = (data.get("coach_notes") or "").strip()
        tournament_results = (data.get("tournament_results") or "").strip()
        matches = data.get("last_20_matches", [])
        matches_played = len(matches) if isinstance(matches, list) else 0

        match_summary_lines = []
        if isinstance(matches, list):
            for m in matches[:20]:
                if not isinstance(m, dict):
                    continue
                line = (
                    f"- {m.get('date', 'Unknown date')}: vs {m.get('opponent', 'Unknown')} | "
                    f"Result: {m.get('result', 'N/A')} | Score: {m.get('score', 'N/A')} | "
                    f"Notes: {m.get('notes', '')}"
                )
                match_summary_lines.append(line)
        match_summary = "\n".join(match_summary_lines) if match_summary_lines else "No recorded matches yet."

        client = Groq(api_key=GROQ_API_KEY)

        system_instruction = (
            "You are an expert US tennis performance analyst generating a short personalized welcome-back briefing "
            "for a returning athlete on the AdvantageAI dashboard. Use the athlete's stored profile and match history "
            "to identify genuine patterns of improvement and ongoing weaknesses, and recommend a single focus for "
            "today's session. Be specific, encouraging, and data-grounded, referencing the athlete by first name in "
            "the welcome message. If match history is sparse, be honest about limited data while still giving useful "
            "direction. "
            "You MUST respond with a valid, clean JSON object matching this exact schema layout without any markdown "
            "blocks outside it:\n"
            "{\n"
            "  \"welcome_message\": \"Welcome back, Ganesh. You've played 26 matches.\",\n"
            "  \"readiness_score\": 78,\n"
            "  \"biggest_improvement\": \"+18% backhand consistency\",\n"
            "  \"current_weakness\": \"Second serve under pressure\",\n"
            "  \"todays_focus\": \"Crosscourt forehands\"\n"
            "}"
        )

        user_payload = (
            f"Player Name: {full_name}\n"
            f"Height: {height}\n"
            f"Play Style: {play_style}\n"
            f"Favorite Serve: {favorite_serve}\n"
            f"Known Weaknesses: {weaknesses}\n"
            f"Recent Injuries: {recent_injuries}\n"
            f"Coach Notes: {coach_notes}\n"
            f"Tournament Results: {tournament_results}\n"
            f"Total Recorded Matches: {matches_played}\n"
            f"Recent Match Log (most recent first):\n{match_summary}"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_content = response.choices[0].message.content
        parsed_result = json.loads(raw_content)
        parsed_result["matches_played"] = matches_played
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({"error": f"Profile insight pipeline failure: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def technical_chat():
    """Processes interactive user queries regarding drills, settings, and general tennis strategies."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"error": "Chat prompt value cannot be empty."}), 400

        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are AdvantageAI, an expert 24/7 technical tennis mentor and strategy guide built for US high school "
                    "and college competitive tennis athletes. Answer follow-up queries with direct tactical solutions, "
                    "mental framework improvements, or mechanical execution suggestions. Keep answers concise and strictly actionable."
                )
            }
        ]
        
        for turn in history:
            messages.append({"role": turn.get("role"), "content": turn.get("content")})
            
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )

        return jsonify({"response": response.choices[0].message.content.strip()})

    except Exception as e:
        return jsonify({"error": f"Interactive chat system breakdown: {str(e)}"}), 500

def _chunked(items, size):
    """Yield successive chunks of `size` items from `items`."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _extract_video_frames(video_path, max_frames=MAX_VIDEO_FRAMES, max_width=MAX_FRAME_WIDTH):
    """Samples up to `max_frames` evenly-spaced frames from the video at `video_path`,
    downsizes them for faster/cheaper vision API calls, and returns a list of dicts:
    {frame_index, timestamp_sec, base64}. Raises ValueError on unreadable input."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video file. Please try a standard MP4/MOV export.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    raw_frames = []  # list of (frame_index, timestamp_sec, frame_array)

    if total_frames > 0:
        span = max(total_frames - 1, 1)
        sample_indices = sorted(set(
            int(round(i * span / max(max_frames - 1, 1))) for i in range(max_frames)
        ))
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                raw_frames.append((idx, idx / fps, frame))
    else:
        # Frame count metadata wasn't available (common with some streamed/odd containers).
        # Fall back to reading sequentially, bounded by a safety cap, then subsample evenly.
        safety_cap = 3000
        sequential_frames = []
        count = 0
        while count < safety_cap:
            success, frame = cap.read()
            if not success:
                break
            sequential_frames.append(frame)
            count += 1
        if sequential_frames:
            step = max(len(sequential_frames) // max_frames, 1)
            for idx in range(0, len(sequential_frames), step):
                if len(raw_frames) >= max_frames:
                    break
                raw_frames.append((idx, idx / fps, sequential_frames[idx]))

    cap.release()

    if not raw_frames:
        raise ValueError("No readable frames were found in the uploaded video.")

    encoded_frames = []
    for frame_index, timestamp_sec, frame in raw_frames:
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            frame = cv2.resize(frame, (max_width, int(height * scale)))
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not success:
            continue
        encoded_frames.append({
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 2),
            "base64": base64.b64encode(buffer).decode('utf-8')
        })

    if not encoded_frames:
        raise ValueError("Video frames were read but could not be encoded for analysis.")

    return encoded_frames


def _analyze_frame_batch(client, batch, player_level, match_context):
    """Sends one batch (<= VISION_BATCH_SIZE) of frames to the Groq vision model and
    returns a list of {shot_type, score, feedback} dicts, one per frame in order."""
    content = [{
        "type": "text",
        "text": (
            f"You are analyzing {len(batch)} sequential still frames captured from an amateur/competitive "
            f"tennis match video, in chronological order. Player competitive level: {player_level}. "
            f"Match context provided by the player: {match_context or 'Not specified'}. "
            "For EACH image, identify the tennis shot or moment being executed by the primary player "
            "(e.g. Forehand, Backhand, Serve, Volley, Overhead Smash, Return of Serve, Footwork/Recovery, "
            "or 'No clear shot visible' if the frame doesn't show an identifiable stroke). "
            "Score the shot's technical execution from 1 to 100, where 100 is textbook-perfect technique "
            "and 1 is a significant technical breakdown. Base the score on visible contact point, balance, "
            "footwork, and follow-through. Give one short, specific, actionable feedback sentence per image. "
            "Respond with ONLY a valid JSON object in this exact schema, with exactly "
            f"{len(batch)} entries in the 'shots' array, in the same order as the images were provided:\n"
            "{\"shots\": [{\"shot_type\": \"Forehand\", \"score\": 72, \"feedback\": \"Contact point slightly "
            "late, causing the ball to sail wide.\"}]}"
        )
    }]
    for item in batch:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{item['base64']}"}
        })

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_completion_tokens=1200
    )
    parsed = json.loads(response.choices[0].message.content)
    shots = parsed.get("shots", [])
    return shots if isinstance(shots, list) else []


@app.route('/api/video-analysis', methods=['POST'])
def video_analysis():
    """Accepts an uploaded match video, samples frames across it, scores each sampled
    shot from 1-100 via the Groq vision model, and returns a statistical breakdown of
    strengths and weaknesses derived from those scored shots."""
    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key is missing on the server configuration layer."}), 500

    if 'video' not in request.files or request.files['video'].filename == '':
        return jsonify({"error": "No video file was included in the upload payload."}), 400

    video_file = request.files['video']
    player_level = (request.form.get('player_level') or 'High School Varsity / NTRP 3.5').strip()
    match_context = (request.form.get('match_context') or '').strip()

    _, ext = os.path.splitext(video_file.filename)
    ext = ext.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported video format '{ext or 'unknown'}'. Please upload MP4, MOV, M4V, AVI, WEBM, or MKV."
        }), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            video_file.save(tmp_file.name)
            tmp_path = tmp_file.name

        frames = _extract_video_frames(tmp_path)

        client = Groq(api_key=GROQ_API_KEY)

        all_shots = []
        for batch in _chunked(frames, VISION_BATCH_SIZE):
            batch_results = _analyze_frame_batch(client, batch, player_level, match_context)
            for offset, frame_meta in enumerate(batch):
                shot_data = batch_results[offset] if offset < len(batch_results) else {}
                score = shot_data.get("score")
                if not isinstance(score, (int, float)):
                    score = None
                all_shots.append({
                    "frame_index": frame_meta["frame_index"],
                    "timestamp_sec": frame_meta["timestamp_sec"],
                    "shot_type": shot_data.get("shot_type", "Unclassified"),
                    "score": score,
                    "feedback": shot_data.get("feedback", "")
                })

        scored_shots = [s for s in all_shots if s["score"] is not None]
        if scored_shots:
            average_score = round(sum(s["score"] for s in scored_shots) / len(scored_shots), 1)
            best_shot = max(scored_shots, key=lambda s: s["score"])
            worst_shot = min(scored_shots, key=lambda s: s["score"])
        else:
            average_score = None
            best_shot = None
            worst_shot = None

        shot_summary_lines = [
            f"- t={s['timestamp_sec']}s | {s['shot_type']} | Score: {s['score']} | {s['feedback']}"
            for s in all_shots
        ]
        shot_summary_text = "\n".join(shot_summary_lines) if shot_summary_lines else "No shots were analyzed."

        summary_system_instruction = (
            "You are an expert US tennis performance analyst. You are given a chronological list of shots "
            "sampled from a match video, each with a 1-100 technical execution score and short feedback. "
            "Identify genuine statistical patterns across the shots: recurring weaknesses, standout strengths, "
            "and one focus recommendation for the next practice session. Be specific and reference the shot "
            "types and scores you were given rather than generic tennis advice. "
            "You MUST respond with a valid, clean JSON object matching this exact schema without any markdown "
            "blocks outside it:\n"
            "{\n"
            "  \"strengths\": [\"Specific strength 1\", \"Specific strength 2\"],\n"
            "  \"weaknesses\": [\"Specific weakness 1\", \"Specific weakness 2\"],\n"
            "  \"summary\": \"Two to three sentence overview of the statistical breakdown.\",\n"
            "  \"focus_recommendation\": \"Single most impactful thing to work on next.\"\n"
            "}"
        )
        summary_user_payload = (
            f"Player Competitive Level: {player_level}\n"
            f"Match Context: {match_context or 'Not specified'}\n"
            f"Average Shot Score: {average_score if average_score is not None else 'N/A'}\n"
            f"Total Shots Analyzed: {len(all_shots)}\n"
            f"Shot-by-Shot Breakdown:\n{shot_summary_text}"
        )

        summary_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": summary_system_instruction},
                {"role": "user", "content": summary_user_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.6
        )
        summary_parsed = json.loads(summary_response.choices[0].message.content)

        return jsonify({
            "shots": all_shots,
            "stats": {
                "total_shots_analyzed": len(all_shots),
                "average_score": average_score,
                "best_shot": best_shot,
                "worst_shot": worst_shot
            },
            "strengths": summary_parsed.get("strengths", []),
            "weaknesses": summary_parsed.get("weaknesses", []),
            "summary": summary_parsed.get("summary", ""),
            "focus_recommendation": summary_parsed.get("focus_recommendation", "")
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Video analysis pipeline failure: {str(e)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.errorhandler(413)
def file_too_large(_e):
    """Returns a clean JSON error when an uploaded video exceeds MAX_CONTENT_LENGTH,
    instead of the default HTML error page."""
    return jsonify({"error": "Uploaded video is too large. Please upload a file under 200 MB."}), 413


if __name__ == '__main__':
    app.run(debug=True)