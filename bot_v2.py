import os
import json
import asyncio
import libsql
import urllib.request
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler, CommandHandler,
    ConversationHandler, filters, ContextTypes
)
from google import genai
from pydantic import BaseModel, Field
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

load_dotenv()

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID"))
except (TypeError, ValueError):
    print("Error: ALLOWED_USER_ID not set properly in .env")
    exit(1)

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

SCOPES = ['https://www.googleapis.com/auth/calendar']

# --- Conversation States ---
(
    AWAITING_CONFIRMATION,
    EDIT_FIELD,
    CHOOSING_SETTING,
    TYPING_SETTING_VALUE,
    CHOOSING_COLOR,
    ADDING_CATEGORY,
    CONFIRM_DELETE_CATEGORY,
    UPDATE_REVIEW,
) = range(8)

COLOR_NAMES = {
    "1": "🟣 Lavender", "2": "🌿 Sage", "3": "🍇 Grape", "4": "🦩 Flamingo",
    "5": "🍌 Banana", "6": "🍊 Tangerine", "7": "🦚 Peacock", "8": "⬛ Graphite",
    "9": "🫐 Blueberry", "10": "🌱 Basil", "11": "🍅 Tomato"
}

COLOR_SHORT = {
    "1": "🟣 Lavender", "2": "🌿 Sage", "3": "🍇 Grape", "4": "🦩 Flamingo",
    "5": "🍌 Banana", "6": "🍊 Tangerine", "7": "🦚 Peacock", "8": "⬛ Graphite",
    "9": "🫐 Blueberry", "10": "🌱 Basil", "11": "🍅 Tomato"
}


# ============================================================
# Singapore Time (via timeapi.io — timezone-safe for Render)
# ============================================================
def get_sg_now() -> datetime:
    """
    Fetch current Singapore time from timeapi.io.
    Falls back to local time + UTC+8 offset if the request fails.
    """
    try:
        url = "https://timeapi.io/api/Time/current/zone?timeZone=Asia/Singapore"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        # data["dateTime"] is like "2026-02-26T12:12:46.1950398"
        dt_str = data["dateTime"].split(".")[0]  # strip sub-seconds
        sg_tz = timezone(timedelta(hours=8))
        return datetime.fromisoformat(dt_str).replace(tzinfo=sg_tz)
    except Exception as e:
        print(f"Warning: timeapi.io failed ({e}), falling back to UTC+8 offset.")
        sg_tz = timezone(timedelta(hours=8))
        return datetime.now(tz=timezone.utc).astimezone(sg_tz)


def get_sg_time_str() -> str:
    """Return a human-readable Singapore time string for prompts."""
    now = get_sg_now()
    return now.strftime("%A, %d %B %Y at %I:%M %p")


def get_sg_iso() -> str:
    """Return current Singapore time as ISO 8601 string."""
    return get_sg_now().isoformat()


# ============================================================
# Config Management (stored in Turso as key-value)
# ============================================================
def _get_db():
    return libsql.connect(
        database=TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN,
    )


def _init_config_table():
    conn = _get_db()
    conn.execute(
        'CREATE TABLE IF NOT EXISTS bot_config '
        '(key TEXT PRIMARY KEY, value TEXT)'
    )
    conn.commit()


def load_config():
    default_config = {
        "reminder_minutes": [30],
        "colors": {"Exam": "11", "Deadline": "10", "Event": "9", "Other": "1"},
        "preferred_model": None,
    }
    try:
        conn = _get_db()
        cur = conn.execute("SELECT value FROM bot_config WHERE key = ?", ("app_config",))
        row = cur.fetchone()
        if row:
            cfg = json.loads(row[0])
            if "reminder_minutes" not in cfg:
                cfg["reminder_minutes"] = [30]
            if "colors" not in cfg:
                cfg["colors"] = default_config["colors"]
            if "preferred_model" not in cfg:
                cfg["preferred_model"] = None
            return cfg
        else:
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"Warning: Could not load config from Turso: {e}")
        save_config(default_config)
        return default_config


def save_config(config_data):
    try:
        conn = _get_db()
        conn.execute(
            'INSERT OR REPLACE INTO bot_config (key, value) VALUES (?, ?)',
            ("app_config", json.dumps(config_data)),
        )
        conn.commit()
    except Exception as e:
        print(f"Warning: Could not save config to Turso: {e}")


def format_time_for_user(iso_string):
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%d %b %Y, %I:%M %p").lstrip("0").replace(" 0", " ")
    except Exception:
        return iso_string


# ============================================================
# Database (Turso)
# ============================================================
def init_db():
    conn = _get_db()
    conn.execute(
        'CREATE TABLE IF NOT EXISTS event_mapping '
        '(message_id INTEGER PRIMARY KEY, google_event_id TEXT)'
    )
    conn.commit()
    _init_config_table()


def save_event_mapping(message_id, google_event_id):
    conn = _get_db()
    conn.execute(
        'INSERT OR REPLACE INTO event_mapping VALUES (?, ?)',
        (message_id, google_event_id),
    )
    conn.commit()


def get_google_event_id(message_id):
    conn = _get_db()
    cur = conn.execute(
        'SELECT google_event_id FROM event_mapping WHERE message_id = ?',
        (message_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


# ============================================================
# Gemini Schema
# ============================================================
class CalendarEvent(BaseModel):
    summary: str = Field(description="The title of the event")
    event_type: str = Field(
        description="The category of the event. Pick the most appropriate one from the list the user provides."
    )
    location: str = Field(description="The location of the event. Leave blank if none.")
    description: str = Field(description="The full body text.")
    start_time: str = Field(
        description="Start time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS+08:00) for Singapore time"
    )
    end_time: str = Field(
        description="End time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS+08:00) for Singapore time"
    )


# ============================================================
# Model Fallback Chain
# ============================================================
MODEL_CHAIN = [
    {"id": "gemini-2.5-flash",         "name": "Gemini 2.5 Flash",     "supports_schema": True},
    {"id": "gemini-2.5-flash-lite",    "name": "Gemini 2.5 Flash Lite","supports_schema": True},
    {"id": "gemini-3-flash-preview",   "name": "Gemini 3 Flash",       "supports_schema": True},
    {"id": "gemma-3-4b-it",            "name": "Gemma 3 4B",           "supports_schema": False},
    {"id": "gemma-3-27b-it",           "name": "Gemma 3 27B",          "supports_schema": False},
    {"id": "gemma-3-12b-it",           "name": "Gemma 3 12B",          "supports_schema": False},
]

_CALENDAR_JSON_TEMPLATE = (
    'You MUST respond with ONLY a valid JSON object (no markdown, no explanation) '
    'using exactly these keys:\n'
    '{"summary": "...", "event_type": "...", "location": "...", '
    '"description": "...", "start_time": "YYYY-MM-DDTHH:MM:SS+08:00", '
    '"end_time": "YYYY-MM-DDTHH:MM:SS+08:00"}'
)


def _get_ordered_chain():
    app_config = load_config()
    preferred_id = app_config.get("preferred_model")
    if not preferred_id:
        return list(MODEL_CHAIN)

    preferred_idx = None
    for i, m in enumerate(MODEL_CHAIN):
        if m["id"] == preferred_id:
            preferred_idx = i
            break

    if preferred_idx is None:
        return list(MODEL_CHAIN)

    chain = [MODEL_CHAIN[preferred_idx]]
    for i, m in enumerate(MODEL_CHAIN):
        if i != preferred_idx:
            chain.append(m)
    return chain


async def generate_with_fallback(
    prompt: str,
    status_message=None,
    use_schema: bool = True,
):
    client = genai.Client()
    last_error = None
    chain = _get_ordered_chain()

    for model_info in chain:
        model_id = model_info["id"]
        model_name = model_info["name"]

        if status_message:
            try:
                await status_message.edit_text(f"⏳ Processing… ({model_name})")
            except Exception:
                pass

        try:
            if use_schema and model_info["supports_schema"]:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': CalendarEvent,
                    },
                )
            else:
                full_prompt = f"{prompt}\n\n{_CALENDAR_JSON_TEMPLATE}"
                config = {}
                if model_info["supports_schema"]:
                    config['response_mime_type'] = 'application/json'
                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt,
                    config=config if config else None,
                )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            return parsed, model_name

        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                continue
            raise

    raise last_error


async def generate_text_with_fallback(
    prompt: str,
    status_message=None,
):
    client = genai.Client()
    last_error = None
    chain = _get_ordered_chain()

    for model_info in chain:
        model_id = model_info["id"]
        model_name = model_info["name"]

        if status_message:
            try:
                await status_message.edit_text(f"⏳ Processing… ({model_name})")
            except Exception:
                pass

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            return response.text.strip(), model_name

        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                continue
            raise

    raise last_error


# ============================================================
# Google Calendar helpers
# ============================================================
def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)


# ============================================================
# Intent Detection & Calendar Query
# ============================================================
QUERY_KEYWORDS = [
    "what do i have", "what's on", "whats on", "do i have anything",
    "show me", "list my", "any events", "schedule for", "agenda",
    "what is scheduled", "am i free", "am i busy", "what time",
    "remind me what", "tmr", "tomorrow", "today", "this week",
    "next week", "upcoming", "calendar", "what's happening", "whats happening",
]


def looks_like_query(text: str) -> bool:
    """Quick heuristic check before calling the LLM."""
    lowered = text.lower()
    return any(kw in lowered for kw in QUERY_KEYWORDS)


async def classify_intent(user_text: str) -> dict:
    """
    Ask Gemini to classify the message intent and extract query parameters.
    Returns dict: { "intent": "query"|"create", "time_min": "...", "time_max": "..." }
    """
    current_time = get_sg_time_str()
    prompt = (
        f"Today is {current_time} (Singapore Time, UTC+8).\n"
        f"Classify the user's message as either a calendar QUERY (asking about existing events) "
        f"or a calendar CREATE/UPDATE request (adding/editing an event).\n\n"
        f"User message: \"{user_text}\"\n\n"
        f"Respond with ONLY valid JSON like:\n"
        f'{{ "intent": "query", "time_min": "YYYY-MM-DDTHH:MM:SS+08:00", "time_max": "YYYY-MM-DDTHH:MM:SS+08:00" }}\n'
        f"or\n"
        f'{{ "intent": "create" }}\n\n'
        f"For queries, set time_min and time_max to cover the relevant period "
        f"(e.g. for 'tomorrow' use start and end of tomorrow, for 'this week' use Mon–Sun). "
        f"Always use UTC+8 offset in timestamps."
    )
    raw, _ = await generate_text_with_fallback(prompt)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def fetch_events(time_min: str, time_max: str) -> list:
    """Fetch events from Google Calendar between time_min and time_max."""
    service = get_calendar_service()
    result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime',
        maxResults=20,
    ).execute()
    return result.get('items', [])


def format_events_for_display(events: list) -> str:
    if not events:
        return "📭 No events found for that period."

    lines = []
    for ev in events:
        start = ev['start'].get('dateTime', ev['start'].get('date', ''))
        try:
            start_display = format_time_for_user(start)
        except Exception:
            start_display = start
        title = ev.get('summary', '(No title)')
        location = ev.get('location', '')
        loc_str = f"\n  📍 {location}" if location else ""
        lines.append(f"• *{title}*\n  🕐 {start_display}{loc_str}")

    return "\n\n".join(lines)


# ============================================================
# /model command — interactive model picker
# ============================================================
async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_config = load_config()
    preferred_id = app_config.get("preferred_model")

    rows = []
    for m in MODEL_CHAIN:
        label = m["name"]
        if m["id"] == preferred_id:
            label = f"✅ {label}"
        rows.append([InlineKeyboardButton(label, callback_data=f"mdl_{m['id']}")])

    auto_label = "✅ 🔄 Auto (best available)" if not preferred_id else "🔄 Auto (best available)"
    rows.append([InlineKeyboardButton(auto_label, callback_data="mdl_auto")])

    current_name = "Auto (best available)"
    if preferred_id:
        for m in MODEL_CHAIN:
            if m["id"] == preferred_id:
                current_name = m["name"]
                break

    await update.message.reply_text(
        f"🤖 *Model Selection*\n\n"
        f"Current: *{current_name}*\n"
        f"Pick a model to use first. If it hits its rate limit, "
        f"the bot will automatically fall back through the remaining models.",
        reply_markup=InlineKeyboardMarkup(rows),
        parse_mode='Markdown',
    )


async def model_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    app_config = load_config()

    if action == "mdl_auto":
        app_config["preferred_model"] = None
        save_config(app_config)
        await query.edit_message_text(
            "✅ Model set to *Auto* — will use the best available model.",
            parse_mode='Markdown',
        )
        return

    if action.startswith("mdl_"):
        chosen_id = action[len("mdl_"):]
        chosen_name = None
        for m in MODEL_CHAIN:
            if m["id"] == chosen_id:
                chosen_name = m["name"]
                break

        if chosen_name:
            app_config["preferred_model"] = chosen_id
            save_config(app_config)
            await query.edit_message_text(
                f"✅ Preferred model set to *{chosen_name}*\n"
                f"_If rate-limited, will auto-fallback to the next available model._",
                parse_mode='Markdown',
            )
        else:
            await query.edit_message_text("❌ Unknown model.")


# ============================================================
# /help command
# ============================================================
HELP_TEXT = (
    "📖 *Available Commands*\n\n"
    "• *Send any text* — I'll extract event details and add them to Google Calendar.\n"
    "• *Ask about your schedule* — e.g. 'what do I have tmr?', 'am I free this week?'\n"
    "• *Reply to a confirmed event* with new details — I'll update the event.\n"
    "• *Reply with* `delete` / `cancel` / `remove` — I'll delete the event.\n\n"
    "*Commands:*\n"
    "/help — Show this help message\n"
    "/config — Open settings (reminders, colors, categories)\n"
    "/model — Choose which AI model to use\n"
    "/cancel — Cancel the current action\n"
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode='Markdown')


# ============================================================
# Event Extraction Flow
# ============================================================
async def start_extraction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.reply_to_message:
        return await handle_update_reply(update, context)

    user_text = update.message.text
    status_message = await update.message.reply_text("⏳ Thinking...")

    # --- Route: query vs. create ---
    try:
        if looks_like_query(user_text):
            intent_data = await classify_intent(user_text)
        else:
            intent_data = {"intent": "create"}
    except Exception as e:
        print(f"Intent classification failed: {e}, defaulting to create.")
        intent_data = {"intent": "create"}

    if intent_data.get("intent") == "query":
        try:
            events = fetch_events(intent_data["time_min"], intent_data["time_max"])
            reply = format_events_for_display(events)
            await status_message.edit_text(reply, parse_mode='Markdown')
        except Exception as e:
            await status_message.edit_text(f"❌ Could not fetch calendar: {e}")
        return ConversationHandler.END

    # --- Original creation flow ---
    app_config = load_config()
    category_list = ", ".join(f"'{c}'" for c in app_config["colors"].keys())
    current_time = get_sg_time_str()

    system_prompt = (
        f"Today's date and time is {current_time} (Singapore Time, UTC+8). "
        f"Use this context to accurately infer missing years, months, or relative days. "
        f"The available event categories are: {category_list}. "
        f"Pick the most appropriate one for 'event_type'. "
        f"Extract the event details from this announcement:\n\n{user_text}"
    )

    try:
        parsed, model_name = await generate_with_fallback(
            system_prompt, status_message=status_message
        )
        context.user_data['draft_event'] = parsed
        context.user_data['last_model'] = model_name

        valid_types = set(app_config["colors"].keys())
        if context.user_data['draft_event'].get('event_type') not in valid_types:
            context.user_data['draft_event']['event_type'] = 'Other'

        await send_draft_menu(status_message, context)
        return AWAITING_CONFIRMATION

    except Exception as e:
        await status_message.edit_text(f"❌ Extraction failed (all models exhausted): {e}")
        return ConversationHandler.END


async def send_draft_menu(message_obj, context):
    event_data = context.user_data['draft_event']
    start_display = format_time_for_user(event_data['start_time'])
    end_display = format_time_for_user(event_data['end_time'])

    keyboard = [
        [InlineKeyboardButton("✅ Confirm & Add", callback_data="act_confirm")],
        [
            InlineKeyboardButton("✏️ Title", callback_data="edit_summary"),
            InlineKeyboardButton("✏️ Type", callback_data="edit_event_type"),
        ],
        [
            InlineKeyboardButton("✏️ Location", callback_data="edit_location"),
            InlineKeyboardButton("✏️ Description", callback_data="edit_description"),
        ],
        [
            InlineKeyboardButton("✏️ Start Time", callback_data="edit_start_time"),
            InlineKeyboardButton("✏️ End Time", callback_data="edit_end_time"),
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data="act_cancel")],
    ]

    draft_msg = (
        f"*📝 Review Draft Event:*\n\n"
        f"*Title:* {event_data['summary']}\n"
        f"*Type:* {event_data.get('event_type', 'Other')}\n"
        f"*Location:* {event_data.get('location') or 'N/A'}\n"
        f"*Description:* {_truncate(event_data.get('description', ''), 120)}\n"
        f"*Starts:* {start_display}\n"
        f"*Ends:* {end_display}\n\n"
        f"Does this look correct?"
    )

    model_name = context.user_data.get('last_model')
    if model_name:
        draft_msg += f"\n_{model_name}_"

    try:
        await message_obj.edit_text(
            draft_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown',
        )
    except AttributeError:
        await message_obj.reply_text(
            draft_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown',
        )


def _truncate(text: str, length: int) -> str:
    if not text:
        return "N/A"
    return text if len(text) <= length else text[:length] + "…"


# ============================================================
# Main Draft Button Handler
# ============================================================
async def main_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "act_cancel":
        await query.edit_message_text("❌ Action cancelled.")
        context.user_data.clear()
        return ConversationHandler.END

    if action == "act_confirm":
        event_data = context.user_data['draft_event']
        app_config = load_config()

        e_type = event_data.get('event_type', 'Other')
        color_id = app_config['colors'].get(e_type, "1")
        reminders = [
            {'method': 'popup', 'minutes': m}
            for m in app_config.get('reminder_minutes', [30])
        ]

        service = get_calendar_service()
        event_body = {
            'summary': event_data['summary'],
            'location': event_data.get('location', ''),
            'description': event_data.get('description', ''),
            'colorId': color_id,
            'reminders': {'useDefault': False, 'overrides': reminders},
            'start': {'dateTime': event_data['start_time'], 'timeZone': 'Asia/Singapore'},
            'end': {'dateTime': event_data['end_time'], 'timeZone': 'Asia/Singapore'},
        }

        try:
            created = service.events().insert(calendarId='primary', body=event_body).execute()
            save_event_mapping(query.message.message_id, created['id'])

            start_display = format_time_for_user(event_data['start_time'])
            end_display = format_time_for_user(event_data['end_time'])

            success_msg = (
                f"✅ *Event Successfully Added!*\n\n"
                f"*Title:* {event_data['summary']}\n"
                f"*Type:* {e_type} _(Color: {COLOR_NAMES.get(color_id, 'Default')})_\n"
                f"*Location:* {event_data.get('location') or 'N/A'}\n"
                f"*Starts:* {start_display}\n"
                f"*Ends:* {end_display}\n\n"
                f"[🗓️ View on Google Calendar]({created.get('htmlLink')})\n\n"
                f"_💡 Reply to this message with 'delete' to remove it, or with new details to update it._"
            )
            await query.edit_message_text(
                success_msg, parse_mode='Markdown', disable_web_page_preview=True
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Failed to add to calendar: {e}")

        context.user_data.clear()
        return ConversationHandler.END

    if action.startswith("edit_"):
        field = action[len("edit_"):]
        context.user_data['editing_field'] = field

        if field == "event_type":
            app_config = load_config()
            categories = list(app_config["colors"].keys())
            buttons = [
                InlineKeyboardButton(cat, callback_data=f"settype_{cat}")
                for cat in categories
            ]
            rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
            rows.append([InlineKeyboardButton("⬅️ Back", callback_data="edit_back")])
            await query.edit_message_text(
                "Choose the event type:",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return AWAITING_CONFIRMATION

        field_labels = {
            "summary": "Title",
            "location": "Location",
            "description": "Description",
            "start_time": "Start Time (e.g. 'tomorrow 3pm' or '2025-07-01 14:00')",
            "end_time": "End Time (e.g. 'tomorrow 5pm' or '2025-07-01 16:00')",
        }
        label = field_labels.get(field, field)
        await query.edit_message_text(f"Please type the new value for *{label}*:", parse_mode='Markdown')
        return EDIT_FIELD

    if action.startswith("settype_"):
        chosen_type = action[len("settype_"):]
        context.user_data['draft_event']['event_type'] = chosen_type
        context.user_data.pop('editing_field', None)
        await send_draft_menu(query.message, context)
        return AWAITING_CONFIRMATION

    if action == "edit_back":
        context.user_data.pop('editing_field', None)
        await send_draft_menu(query.message, context)
        return AWAITING_CONFIRMATION

    return AWAITING_CONFIRMATION


async def handle_edit_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    field = context.user_data.get('editing_field')
    new_value = update.message.text.strip()

    if not field:
        return AWAITING_CONFIRMATION

    if field in ('start_time', 'end_time'):
        status = await update.message.reply_text("⏳ Parsing time…")
        try:
            current_time = get_sg_time_str()
            prompt = (
                f"Today is {current_time} (Singapore Time, UTC+8). "
                f"Convert this to ISO 8601 format (YYYY-MM-DDTHH:MM:SS+08:00): '{new_value}'. "
                f"Reply with ONLY the ISO 8601 string and nothing else."
            )
            resp_text, _ = await generate_text_with_fallback(prompt, status_message=status)
            parsed = resp_text.strip().strip('"').strip("'")
            datetime.fromisoformat(parsed)
            context.user_data['draft_event'][field] = parsed
            await status.edit_text(f"✅ Parsed as: {format_time_for_user(parsed)}")
        except Exception:
            await status.edit_text("❌ Couldn't parse that time. Please try again (e.g. 'tomorrow 3pm').")
            return EDIT_FIELD
    else:
        context.user_data['draft_event'][field] = new_value

    context.user_data.pop('editing_field', None)

    status_msg = await update.message.reply_text("🔄 Updating draft…")
    await send_draft_menu(status_msg, context)
    return AWAITING_CONFIRMATION


# ============================================================
# Handle Update & Delete via Reply
# ============================================================
async def handle_update_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    replied_msg_id = update.message.reply_to_message.message_id
    google_event_id = get_google_event_id(replied_msg_id)

    if not google_event_id:
        await update.message.reply_text("❌ No calendar record found for that message.")
        return ConversationHandler.END

    new_instructions = update.message.text.strip()

    if new_instructions.lower() in ('delete', 'cancel', 'remove'):
        keyboard = [
            [
                InlineKeyboardButton("🗑️ Yes, delete", callback_data="del_yes"),
                InlineKeyboardButton("⬅️ No, keep it", callback_data="del_no"),
            ]
        ]
        context.user_data['delete_event_id'] = google_event_id
        await update.message.reply_text(
            "⚠️ *Are you sure you want to delete this event?*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown',
        )
        return UPDATE_REVIEW

    # --- Update with review ---
    status_message = await update.message.reply_text("⏳ Fetching and updating the event…")
    try:
        service = get_calendar_service()
        existing_event = service.events().get(
            calendarId='primary', eventId=google_event_id
        ).execute()

        app_config = load_config()
        category_list = ", ".join(f"'{c}'" for c in app_config["colors"].keys())

        prompt = (
            f"Here is a calendar event: {json.dumps(existing_event)}. "
            f"The available categories are: {category_list}. "
            f"The user wants to make this update: '{new_instructions}'. "
            f"Return the complete updated details."
        )
        updated_data, model_name = await generate_with_fallback(
            prompt, status_message=status_message
        )

        context.user_data['draft_event'] = updated_data
        context.user_data['update_event_id'] = google_event_id
        context.user_data['update_replied_msg_id'] = replied_msg_id

        start_display = format_time_for_user(updated_data['start_time'])
        end_display = format_time_for_user(updated_data['end_time'])

        keyboard = [
            [InlineKeyboardButton("✅ Apply Update", callback_data="upd_confirm")],
            [InlineKeyboardButton("❌ Cancel", callback_data="upd_cancel")],
        ]

        msg = (
            f"*📝 Review Updated Event:*\n\n"
            f"*Title:* {updated_data['summary']}\n"
            f"*Type:* {updated_data.get('event_type', 'Other')}\n"
            f"*Location:* {updated_data.get('location') or 'N/A'}\n"
            f"*Starts:* {start_display}\n"
            f"*Ends:* {end_display}\n\n"
            f"Apply this update?\n_{model_name}_"
        )
        await status_message.edit_text(
            msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown'
        )
        return UPDATE_REVIEW

    except Exception as e:
        await status_message.edit_text(f"❌ Update failed (all models exhausted): {e}")
        return ConversationHandler.END


async def update_review_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "del_yes":
        event_id = context.user_data.get('delete_event_id')
        try:
            service = get_calendar_service()
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            await query.edit_message_text(
                "✅ *Event permanently deleted from your calendar.*", parse_mode='Markdown'
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Deletion failed: {e}")
        context.user_data.clear()
        return ConversationHandler.END

    if action == "del_no":
        await query.edit_message_text("👍 Event kept. No changes made.")
        context.user_data.clear()
        return ConversationHandler.END

    if action == "upd_confirm":
        event_id = context.user_data.get('update_event_id')
        updated_data = context.user_data.get('draft_event', {})
        replied_msg_id = context.user_data.get('update_replied_msg_id')

        app_config = load_config()
        e_type = updated_data.get('event_type', 'Other')
        color_id = app_config['colors'].get(e_type, "1")

        patch_body = {
            'summary': updated_data['summary'],
            'location': updated_data.get('location', ''),
            'description': updated_data.get('description', ''),
            'colorId': color_id,
            'start': {'dateTime': updated_data['start_time'], 'timeZone': 'Asia/Singapore'},
            'end': {'dateTime': updated_data['end_time'], 'timeZone': 'Asia/Singapore'},
        }
        try:
            service = get_calendar_service()
            patched = service.events().patch(
                calendarId='primary', eventId=event_id, body=patch_body
            ).execute()

            save_event_mapping(query.message.message_id, event_id)

            start_display = format_time_for_user(updated_data['start_time'])
            end_display = format_time_for_user(updated_data['end_time'])

            msg = (
                f"✅ *Event Updated!*\n\n"
                f"*Title:* {updated_data['summary']}\n"
                f"*Type:* {e_type} _(Color: {COLOR_NAMES.get(color_id, 'Default')})_\n"
                f"*Location:* {updated_data.get('location') or 'N/A'}\n"
                f"*Starts:* {start_display}\n"
                f"*Ends:* {end_display}\n\n"
                f"[🗓️ View on Google Calendar]({patched.get('htmlLink')})\n\n"
                f"_💡 Reply to this message to update or delete it again._"
            )
            await query.edit_message_text(
                msg, parse_mode='Markdown', disable_web_page_preview=True
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Update failed: {e}")

        context.user_data.clear()
        return ConversationHandler.END

    if action == "upd_cancel":
        await query.edit_message_text("❌ Update cancelled. Event unchanged.")
        context.user_data.clear()
        return ConversationHandler.END

    return UPDATE_REVIEW


# ============================================================
# Configuration Menu
# ============================================================
def _build_config_keyboard(app_config):
    c = app_config['colors']
    rows = [
        [InlineKeyboardButton(
            f"⏰ Reminders: {app_config['reminder_minutes']} min",
            callback_data="cfg_reminders",
        )],
    ]
    for cat_name, color_id in c.items():
        color_label = COLOR_NAMES.get(color_id, f"ID {color_id}")
        rows.append([InlineKeyboardButton(
            f"{color_label}  —  {cat_name}",
            callback_data=f"cfg_color_{cat_name}",
        )])
    rows.append([
        InlineKeyboardButton("➕ Add Category", callback_data="cfg_add_cat"),
        InlineKeyboardButton("➖ Remove Category", callback_data="cfg_del_cat"),
    ])
    rows.append([InlineKeyboardButton("✅ Done", callback_data="cfg_done")])
    return InlineKeyboardMarkup(rows)


async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_config = load_config()
    msg = "⚙️ *Bot Settings*\nTap a button to modify it:"
    await update.message.reply_text(
        msg,
        reply_markup=_build_config_keyboard(app_config),
        parse_mode='Markdown',
    )
    return CHOOSING_SETTING


async def _show_config_menu(query, context):
    app_config = load_config()
    msg = "⚙️ *Bot Settings*\nTap a button to modify it:"
    await query.edit_message_text(
        msg,
        reply_markup=_build_config_keyboard(app_config),
        parse_mode='Markdown',
    )
    return CHOOSING_SETTING


async def config_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "cfg_done":
        await query.edit_message_text("✅ Settings saved. Configuration closed.")
        context.user_data.clear()
        return ConversationHandler.END

    if action == "cfg_reminders":
        context.user_data['cfg_key'] = 'reminders'
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="cfg_back")]]
        await query.edit_message_text(
            "Enter new reminder times in minutes.\n"
            "Examples: `30` or `30, 1440` (30 min + 1 day before):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return TYPING_SETTING_VALUE

    if action.startswith("cfg_color_"):
        evt_type = action[len("cfg_color_"):]
        context.user_data['cfg_color_target'] = evt_type
        return await _show_color_picker(query, evt_type)

    if action == "cfg_add_cat":
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="cfg_back")]]
        await query.edit_message_text(
            "Type the name for the new category (e.g. `Meeting`, `Lab`):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return ADDING_CATEGORY

    if action == "cfg_del_cat":
        app_config = load_config()
        protected = {"Other"}
        deletable = [c for c in app_config['colors'] if c not in protected]
        if not deletable:
            await query.answer("No categories to remove (only 'Other' left).", show_alert=True)
            return CHOOSING_SETTING
        buttons = [
            InlineKeyboardButton(cat, callback_data=f"delcat_{cat}")
            for cat in deletable
        ]
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="cfg_back")])
        await query.edit_message_text(
            "Which category do you want to remove?",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return CONFIRM_DELETE_CATEGORY

    if action == "cfg_back":
        return await _show_config_menu(query, context)

    return CHOOSING_SETTING


# --- Interactive Color Picker ---
async def _show_color_picker(query, evt_type):
    buttons = [
        InlineKeyboardButton(COLOR_SHORT[cid], callback_data=f"pickclr_{cid}")
        for cid in sorted(COLOR_SHORT, key=int)
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="cfg_back")])

    await query.edit_message_text(
        f"Pick a color for *{evt_type}*:",
        reply_markup=InlineKeyboardMarkup(rows),
        parse_mode='Markdown',
    )
    return CHOOSING_COLOR


async def color_pick_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "cfg_back":
        return await _show_config_menu(query, context)

    if action.startswith("pickclr_"):
        color_id = action[len("pickclr_"):]
        evt_type = context.user_data.get('cfg_color_target')

        app_config = load_config()
        if evt_type and evt_type in app_config['colors']:
            app_config['colors'][evt_type] = color_id
            save_config(app_config)

        context.user_data.pop('cfg_color_target', None)
        await query.answer(f"✅ {evt_type} → {COLOR_NAMES.get(color_id, color_id)}", show_alert=True)
        return await _show_config_menu(query, context)

    return CHOOSING_COLOR


# --- Add / Remove Category ---
async def handle_add_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.message.text.strip()
    if not name or len(name) > 30:
        await update.message.reply_text("❌ Name must be 1–30 characters. Try /config again.")
        return ConversationHandler.END

    app_config = load_config()
    if name in app_config['colors']:
        await update.message.reply_text(f"❌ Category '{name}' already exists. Try /config again.")
        return ConversationHandler.END

    app_config['colors'][name] = "1"
    save_config(app_config)

    await update.message.reply_text(
        f"✅ Category *{name}* added (default color: {COLOR_NAMES['1']}).\n"
        f"Use /config to change its color.",
        parse_mode='Markdown',
    )
    return ConversationHandler.END


async def confirm_delete_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "cfg_back":
        return await _show_config_menu(query, context)

    if action.startswith("delcat_"):
        cat_name = action[len("delcat_"):]
        app_config = load_config()
        if cat_name in app_config['colors'] and cat_name != "Other":
            del app_config['colors'][cat_name]
            save_config(app_config)
            await query.answer(f"🗑️ '{cat_name}' removed.", show_alert=True)
        else:
            await query.answer("Cannot remove that category.", show_alert=True)

        return await _show_config_menu(query, context)

    return CHOOSING_SETTING


# --- Reminder text input ---
async def handle_config_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    val = update.message.text.strip()
    cfg_key = context.user_data.get('cfg_key')
    app_config = load_config()

    if cfg_key == 'reminders':
        try:
            m_list = [int(x.strip()) for x in val.split(',')]
            if any(m < 0 for m in m_list):
                raise ValueError
            app_config['reminder_minutes'] = m_list
        except ValueError:
            await update.message.reply_text(
                "❌ Invalid format. Use comma-separated positive numbers (e.g. `30, 1440`). Try /config again.",
                parse_mode='Markdown',
            )
            return ConversationHandler.END

    save_config(app_config)
    await update.message.reply_text("✅ Setting updated! Use /config to see your settings.")
    context.user_data.clear()
    return ConversationHandler.END


# ============================================================
# Cancel fallback
# ============================================================
async def cancel_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled current action.")
    context.user_data.clear()
    return ConversationHandler.END


# ============================================================
# Main
# ============================================================
def main():
    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    user_filter = filters.User(user_id=ALLOWED_USER_ID)

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("config", config_command, filters=user_filter),
            MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, start_extraction),
        ],
        states={
            AWAITING_CONFIRMATION: [
                CallbackQueryHandler(main_button_handler, pattern="^(act_|edit_|settype_)"),
            ],
            EDIT_FIELD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, handle_edit_input),
            ],
            UPDATE_REVIEW: [
                CallbackQueryHandler(update_review_handler, pattern="^(upd_|del_)"),
            ],
            CHOOSING_SETTING: [
                CallbackQueryHandler(config_button_handler, pattern="^cfg_"),
                CallbackQueryHandler(confirm_delete_category, pattern="^delcat_"),
            ],
            CHOOSING_COLOR: [
                CallbackQueryHandler(color_pick_handler, pattern="^(pickclr_|cfg_back)"),
            ],
            TYPING_SETTING_VALUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, handle_config_input),
                CallbackQueryHandler(config_button_handler, pattern="^cfg_back"),
            ],
            ADDING_CATEGORY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, handle_add_category),
                CallbackQueryHandler(config_button_handler, pattern="^cfg_back"),
            ],
            CONFIRM_DELETE_CATEGORY: [
                CallbackQueryHandler(confirm_delete_category, pattern="^(delcat_|cfg_back)"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_action, filters=user_filter)],
        per_message=False,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command, filters=user_filter))
    app.add_handler(CommandHandler("model", model_command, filters=user_filter))
    app.add_handler(CallbackQueryHandler(model_button_handler, pattern="^mdl_"))

    # --- Webhook vs Polling Logic ---
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    PORT = int(os.getenv("PORT", "10000"))

    if WEBHOOK_URL:
        print(f"Starting webhook on port {PORT}...")

        async def main_async():
            async with app:
                await app.start()
                await app.bot.set_webhook(url=WEBHOOK_URL + "/webhook")

                from starlette.applications import Starlette
                from starlette.responses import PlainTextResponse, Response
                from starlette.requests import Request
                from starlette.routing import Route
                import uvicorn

                async def health(request: Request):
                    return PlainTextResponse("OK")

                async def telegram_webhook(request: Request):
                    data = await request.json()
                    update = Update.de_json(data=data, bot=app.bot)
                    await app.process_update(update)
                    return Response(status_code=200)

                starlette_app = Starlette(
                    routes=[
                        Route("/", health),
                        Route("/health", health),
                        Route("/webhook", telegram_webhook, methods=["POST"]),
                    ]
                )

                webserver = uvicorn.Server(
                    config=uvicorn.Config(
                        app=starlette_app,
                        host="0.0.0.0",
                        port=PORT,
                        log_level="info",
                    )
                )

                await webserver.serve()
                await app.stop()

        asyncio.run(main_async())
    else:
        print("Bot is fully operational. Awaiting your commands (Local Polling).")
        app.run_polling()


if __name__ == '__main__':
    main()