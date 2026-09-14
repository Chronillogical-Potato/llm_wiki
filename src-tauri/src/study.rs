use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

const QUESTIONS_PATH: &str = "study/questions.jsonl";
const LEARNING_DIRECTORY: &str = ".llm-wiki/learning";
const EVENTS_PATH: &str = ".llm-wiki/learning/events/events.jsonl";
const QUARANTINE_PATH: &str = ".llm-wiki/learning/quarantine/malformed-events.jsonl";
const MAX_QUESTIONS: usize = 10_000;
const MAX_QUESTION_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVENT_LOG_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JSONL_LINE_BYTES: usize = 64 * 1024;
const MAX_RATIONALE_CHARS: usize = 400;
const MAX_IDEMPOTENCY_CHARS: usize = 128;
const MAX_SESSION_CHARS: usize = 128;
const MAX_RESPONSE_MS: u64 = 6 * 60 * 60 * 1_000;

static EVENT_LOG_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Debug)]
pub struct StudyError {
    pub status: u16,
    pub message: String,
}

impl StudyError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: 409,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
        }
    }

    fn insufficient_storage(message: impl Into<String>) -> Self {
        Self {
            status: 507,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ObjectiveRecord {
    id: String,
    label: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct OptionRecord {
    id: String,
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuestionRecord {
    id: String,
    version: u32,
    title: String,
    objective: ObjectiveRecord,
    concept_ids: Vec<String>,
    item_type: String,
    prompt: String,
    options: Vec<OptionRecord>,
    correct_option: String,
    explanation: String,
    confusable: Value,
    #[serde(default)]
    remediation: Vec<Value>,
    #[serde(default)]
    source_refs: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AttemptInput {
    question_id: String,
    question_version: u32,
    selected_response: String,
    confidence_before: u8,
    #[serde(default)]
    rationale_text: Option<String>,
    evidence_modality: String,
    response_ms: u64,
    session_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvidenceState {
    state: String,
    summary: String,
    evidence_count: usize,
    modalities: Vec<String>,
    unresolved_misconceptions: Vec<String>,
    last_success: Option<String>,
    last_failure: Option<String>,
    next_review: Option<String>,
    event_refs: Vec<String>,
}

#[derive(Debug)]
struct MalformedEvent {
    line: usize,
    raw: String,
    error: String,
}

pub fn next_questions(project_root: &Path, limit: usize) -> Result<Value, StudyError> {
    let root = canonical_project_root(project_root)?;
    let questions = load_questions(&root)?;
    let events = load_events(&root)?;
    let attempt_index = index_attempts(&events);
    let now = Utc::now();

    let mut queue: Vec<(bool, Option<DateTime<Utc>>, usize, Value)> = questions
        .iter()
        .enumerate()
        .map(|(index, question)| {
            let attempts = attempt_index
                .get(&question.id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let latest = attempts.last().copied();
            let due_at = latest
                .and_then(|event| event.get("scheduled_review_at"))
                .and_then(Value::as_str)
                .and_then(parse_timestamp);
            let is_due = latest.is_none() || due_at.map(|value| value <= now).unwrap_or(true);
            let state = evidence_state(attempts, now);
            let reason = if latest.is_none() {
                "Unseen evidence".to_string()
            } else if is_due {
                "Review due".to_string()
            } else if state.state == "fragile" {
                "Mechanism repair scheduled".to_string()
            } else {
                format!(
                    "Scheduled {}",
                    due_at
                        .map(|value| value.to_rfc3339())
                        .unwrap_or_else(|| "after more evidence".to_string())
                )
            };
            (
                is_due,
                due_at,
                index,
                public_question(question, &state, &reason, is_due),
            )
        })
        .collect();

    queue.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let questions: Vec<Value> = queue
        .into_iter()
        .take(limit.clamp(1, 50))
        .map(|entry| entry.3)
        .collect();

    Ok(json!({
        "ok": true,
        "questions": questions,
        "generatedAt": now.to_rfc3339(),
    }))
}

pub fn record_attempt(
    project_root: &Path,
    body: &str,
    idempotency_key: &str,
) -> Result<Value, StudyError> {
    validate_idempotency_key(idempotency_key)?;
    let input: AttemptInput = serde_json::from_str(body)
        .map_err(|error| StudyError::bad_request(format!("Invalid learner event: {error}")))?;
    validate_attempt(&input)?;

    let root = canonical_project_root(project_root)?;
    let questions = load_questions(&root)?;
    let question = questions
        .iter()
        .find(|question| question.id == input.question_id)
        .ok_or_else(|| StudyError::not_found("Unknown study question"))?;
    if question.version != input.question_version {
        return Err(StudyError::conflict(format!(
            "Question version {} is current; received {}",
            question.version, input.question_version
        )));
    }
    if !question
        .options
        .iter()
        .any(|option| option.id == input.selected_response)
    {
        return Err(StudyError::bad_request(
            "Selected response is not a question option",
        ));
    }

    let lock = EVENT_LOG_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock
        .lock()
        .map_err(|_| StudyError::internal("Learner event log lock is unavailable"))?;
    let events = load_events(&root)?;
    if let Some(existing) = events
        .iter()
        .find(|event| event.get("idempotency_key").and_then(Value::as_str) == Some(idempotency_key))
    {
        if existing.get("question_id").and_then(Value::as_str) != Some(input.question_id.as_str())
            || existing.get("selected_response").and_then(Value::as_str)
                != Some(input.selected_response.as_str())
        {
            return Err(StudyError::conflict(
                "Idempotency-Key was already used for a different learner event",
            ));
        }
        return Ok(attempt_response(existing.clone(), question));
    }

    let prior_attempts = attempts_for(&events, &question.id);
    let prior_correct_count = prior_attempts
        .iter()
        .filter(|event| event.get("correct").and_then(Value::as_bool) == Some(true))
        .count();
    let correct = input.selected_response == question.correct_option;
    let high_confidence_miss = !correct && input.confidence_before >= 2;
    let now = Utc::now();
    let scheduled_review_at =
        schedule_review(correct, input.confidence_before, prior_correct_count, now);
    let next_reason = next_reason(correct, input.confidence_before, prior_correct_count);
    let error_class = if correct {
        "clean"
    } else if high_confidence_miss {
        "concept_confusion"
    } else {
        "knowledge_gap"
    };
    let rationale = input
        .rationale_text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let event = json!({
        "schema_version": "1.0.0",
        "event_id": Uuid::new_v4().to_string(),
        "idempotency_key": idempotency_key,
        "occurred_at": now.to_rfc3339(),
        "session_id": input.session_id,
        "event_type": "question_attempt",
        "question_id": question.id,
        "question_version": question.version,
        "concept_ids": question.concept_ids,
        "objective_ids": [question.objective.id],
        "selected_response": input.selected_response,
        "correct": correct,
        "confidence_before": input.confidence_before,
        "response_ms": input.response_ms,
        "rationale_text": rationale,
        "error_class": error_class,
        "misconception_ids": if correct { Vec::<String>::new() } else { vec![format!("{}:unresolved", question.concept_ids.first().cloned().unwrap_or_else(|| "unknown".to_string()))] },
        "evidence_modality": input.evidence_modality,
        "source": "study_api",
        "scheduled_review_at": scheduled_review_at.to_rfc3339(),
        "next_reason": next_reason,
    });

    append_event(&root, &event)?;
    Ok(attempt_response(event, question))
}

pub fn learner_state(project_root: &Path, concept_id: Option<&str>) -> Result<Value, StudyError> {
    let root = canonical_project_root(project_root)?;
    let questions = load_questions(&root)?;
    let events = load_events(&root)?;
    let attempt_index = index_attempts(&events);
    let now = Utc::now();
    let states: Vec<Value> = questions
        .iter()
        .filter(|question| {
            concept_id
                .map(|concept| question.concept_ids.iter().any(|item| item == concept))
                .unwrap_or(true)
        })
        .map(|question| {
            let attempts = attempt_index
                .get(&question.id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            json!({
                "questionId": question.id,
                "conceptIds": question.concept_ids,
                "objectiveIds": [question.objective.id],
                "evidence": evidence_state(attempts, now),
            })
        })
        .collect();

    Ok(json!({
        "ok": true,
        "conceptId": concept_id,
        "states": states,
        "supportingEventCount": events.iter().filter(|event| event.get("event_type").and_then(Value::as_str) == Some("question_attempt")).count(),
        "derivedAt": now.to_rfc3339(),
    }))
}

fn canonical_project_root(project_root: &Path) -> Result<PathBuf, StudyError> {
    let root = fs::canonicalize(project_root).map_err(|error| {
        StudyError::internal(format!("Failed to resolve project root: {error}"))
    })?;
    if !root.is_dir() {
        return Err(StudyError::internal(
            "Registered project root is not a directory",
        ));
    }
    Ok(root)
}

fn load_questions(root: &Path) -> Result<Vec<QuestionRecord>, StudyError> {
    let path = safe_existing_file(root, QUESTIONS_PATH)?;
    let metadata = fs::metadata(&path).map_err(|error| {
        StudyError::internal(format!("Failed to inspect study questions: {error}"))
    })?;
    if metadata.len() > MAX_QUESTION_FILE_BYTES {
        return Err(StudyError::internal(
            "Study question file exceeds the 16 MiB safety limit",
        ));
    }
    let file = File::open(&path).map_err(|error| {
        StudyError::internal(format!("Failed to read study questions: {error}"))
    })?;
    let mut reader = BufReader::new(file);
    let mut questions = Vec::new();
    let mut ids = BTreeSet::new();
    let mut line_number = 0;
    while let Some(line) = read_bounded_line(&mut reader, "question", line_number + 1)? {
        line_number += 1;
        if line.trim().is_empty() {
            continue;
        }
        let question: QuestionRecord = serde_json::from_str(&line).map_err(|error| {
            StudyError::internal(format!(
                "Invalid study question line {}: {error}",
                line_number
            ))
        })?;
        if question.options.len() < 2
            || !question
                .options
                .iter()
                .any(|option| option.id == question.correct_option)
        {
            return Err(StudyError::internal(format!(
                "Study question {} has an invalid answer definition",
                question.id
            )));
        }
        if !ids.insert(question.id.clone()) {
            return Err(StudyError::internal(format!(
                "Duplicate study question id: {}",
                question.id
            )));
        }
        questions.push(question);
        if questions.len() > MAX_QUESTIONS {
            return Err(StudyError::internal(
                "Study question file exceeds the safe limit",
            ));
        }
    }
    Ok(questions)
}

fn load_events(root: &Path) -> Result<Vec<Value>, StudyError> {
    let path = root.join(EVENTS_PATH);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(StudyError::internal(format!(
                "Failed to inspect learner event log: {error}"
            )))
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(StudyError::internal(
            "Learner event log must be a regular file inside the registered project",
        ));
    }
    if metadata.len() > MAX_EVENT_LOG_BYTES {
        return Err(StudyError::insufficient_storage(
            "Learner event log exceeds the 64 MiB safety limit",
        ));
    }
    let canonical = fs::canonicalize(&path)
        .map_err(|error| StudyError::internal(format!("Failed to resolve event log: {error}")))?;
    if !canonical.starts_with(root) {
        return Err(StudyError::internal(
            "Learner event log escapes the project root",
        ));
    }

    let file = File::open(&canonical)
        .map_err(|error| StudyError::internal(format!("Failed to read learner events: {error}")))?;
    let mut reader = BufReader::new(file);
    let mut events = Vec::new();
    let mut malformed = Vec::new();
    let mut line_number = 0;
    while let Some(raw) = read_bounded_line(&mut reader, "learner event", line_number + 1)? {
        line_number += 1;
        if raw.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&raw) {
            Ok(value) if value.is_object() => events.push(value),
            Ok(_) => malformed.push(MalformedEvent {
                line: line_number,
                raw,
                error: "event must be a JSON object".to_string(),
            }),
            Err(error) => malformed.push(MalformedEvent {
                line: line_number,
                raw,
                error: error.to_string(),
            }),
        }
    }
    write_quarantine(root, &malformed)?;
    Ok(events)
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    label: &str,
    line_number: usize,
) -> Result<Option<String>, StudyError> {
    let mut bytes = Vec::with_capacity(8 * 1024);
    let mut limited = reader.take((MAX_JSONL_LINE_BYTES + 2) as u64);
    let read = limited.read_until(b'\n', &mut bytes).map_err(|error| {
        StudyError::internal(format!(
            "Failed to read {label} line {line_number}: {error}"
        ))
    })?;
    if read == 0 {
        return Ok(None);
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    }
    if bytes.len() > MAX_JSONL_LINE_BYTES {
        return Err(StudyError::internal(format!(
            "{label} line {line_number} exceeds the 64 KiB safety limit"
        )));
    }
    String::from_utf8(bytes).map(Some).map_err(|error| {
        StudyError::internal(format!("{label} line {line_number} is not UTF-8: {error}"))
    })
}

fn append_event(root: &Path, event: &Value) -> Result<(), StudyError> {
    let directory = ensure_private_directory(root, ".llm-wiki/learning/events")?;
    let path = directory.join("events.jsonl");
    let serialized = serde_json::to_string(event).map_err(|error| {
        StudyError::internal(format!("Failed to serialize learner event: {error}"))
    })?;
    if serialized.len() > MAX_JSONL_LINE_BYTES {
        return Err(StudyError::bad_request(
            "Learner event exceeds the 64 KiB line limit",
        ));
    }
    let mut current_bytes = 0_u64;
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(StudyError::internal(
                "Refusing to append to a non-regular learner event log",
            ));
        }
        current_bytes = metadata.len();
    }
    if current_bytes.saturating_add(serialized.len() as u64 + 1) > MAX_EVENT_LOG_BYTES {
        return Err(StudyError::insufficient_storage(
            "Learner event log reached the 64 MiB safety limit; archive it before recording more evidence",
        ));
    }
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).map_err(|error| {
        StudyError::internal(format!("Failed to open learner event log: {error}"))
    })?;
    file.write_all(serialized.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_data())
        .map_err(|error| StudyError::internal(format!("Failed to append learner event: {error}")))
}

fn write_quarantine(root: &Path, malformed: &[MalformedEvent]) -> Result<(), StudyError> {
    let path = root.join(QUARANTINE_PATH);
    if malformed.is_empty() {
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if metadata.file_type().is_symlink() {
                return Err(StudyError::internal(
                    "Malformed-event quarantine is a symlink",
                ));
            }
            fs::remove_file(&path).map_err(|error| {
                StudyError::internal(format!(
                    "Failed to clear malformed-event quarantine: {error}"
                ))
            })?;
        }
        return Ok(());
    }

    let directory = ensure_private_directory(root, ".llm-wiki/learning/quarantine")?;
    let temporary = directory.join(format!("malformed-events-{}.tmp", Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(|error| {
        StudyError::internal(format!(
            "Failed to create malformed-event quarantine: {error}"
        ))
    })?;
    for item in malformed {
        let line = json!({ "line": item.line, "raw": item.raw, "error": item.error });
        writeln!(file, "{line}").map_err(|error| {
            StudyError::internal(format!(
                "Failed to write malformed-event quarantine: {error}"
            ))
        })?;
    }
    file.sync_all().map_err(|error| {
        StudyError::internal(format!(
            "Failed to sync malformed-event quarantine: {error}"
        ))
    })?;
    fs::rename(&temporary, &path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        StudyError::internal(format!(
            "Failed to publish malformed-event quarantine: {error}"
        ))
    })
}

fn safe_existing_file(root: &Path, relative: &str) -> Result<PathBuf, StudyError> {
    let candidate = relative_path(root, relative)?;
    let metadata = fs::symlink_metadata(&candidate).map_err(|error| {
        StudyError::internal(format!(
            "Required project file {relative} is unavailable: {error}"
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(StudyError::internal(format!(
            "Required project file {relative} must be a regular file"
        )));
    }
    let canonical = fs::canonicalize(&candidate).map_err(|error| {
        StudyError::internal(format!(
            "Failed to resolve project file {relative}: {error}"
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(StudyError::internal(format!(
            "Project file {relative} escapes the registered root"
        )));
    }
    Ok(canonical)
}

fn ensure_private_directory(root: &Path, relative: &str) -> Result<PathBuf, StudyError> {
    let mut current = root.to_path_buf();
    for component in Path::new(relative).components() {
        let Component::Normal(part) = component else {
            return Err(StudyError::internal("Invalid project state directory"));
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(StudyError::internal(
                    "Project state directory contains a symlink or non-directory component",
                ))
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current).map_err(|error| {
                    StudyError::internal(format!(
                        "Failed to create project state directory: {error}"
                    ))
                })?;
                set_private_directory_permissions(&current)?;
            }
            Err(error) => {
                return Err(StudyError::internal(format!(
                    "Failed to inspect project state directory: {error}"
                )))
            }
        }
    }
    let canonical = fs::canonicalize(&current).map_err(|error| {
        StudyError::internal(format!(
            "Failed to resolve project state directory: {error}"
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(StudyError::internal(
            "Project state directory escapes the project root",
        ));
    }
    Ok(canonical)
}

fn relative_path(root: &Path, relative: &str) -> Result<PathBuf, StudyError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(StudyError::internal("Invalid project-relative path"));
    }
    Ok(root.join(path))
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> Result<(), StudyError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
        StudyError::internal(format!(
            "Failed to protect project state directory: {error}"
        ))
    })
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> Result<(), StudyError> {
    Ok(())
}

fn validate_idempotency_key(value: &str) -> Result<(), StudyError> {
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > MAX_IDEMPOTENCY_CHARS
        || !value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | ':' | '.')
        })
    {
        return Err(StudyError::bad_request(
            "Idempotency-Key must be 1-128 safe ASCII characters",
        ));
    }
    Ok(())
}

fn validate_attempt(input: &AttemptInput) -> Result<(), StudyError> {
    if input.question_id.trim().is_empty() || input.question_id.chars().count() > 160 {
        return Err(StudyError::bad_request("questionId is invalid"));
    }
    if input.session_id.trim().is_empty() || input.session_id.chars().count() > MAX_SESSION_CHARS {
        return Err(StudyError::bad_request("sessionId is invalid"));
    }
    if input.confidence_before > 3 {
        return Err(StudyError::bad_request(
            "confidenceBefore must be between 0 and 3",
        ));
    }
    if input.response_ms > MAX_RESPONSE_MS {
        return Err(StudyError::bad_request(
            "responseMs exceeds the six-hour limit",
        ));
    }
    if input
        .rationale_text
        .as_deref()
        .map(|value| value.chars().count() > MAX_RATIONALE_CHARS)
        .unwrap_or(false)
    {
        return Err(StudyError::bad_request(
            "rationaleText exceeds 400 characters",
        ));
    }
    if !matches!(
        input.evidence_modality.as_str(),
        "multiple_choice" | "scenario" | "discrimination" | "pbq" | "teach_back" | "lab"
    ) {
        return Err(StudyError::bad_request("evidenceModality is unsupported"));
    }
    Ok(())
}

fn public_question(
    question: &QuestionRecord,
    state: &EvidenceState,
    reason: &str,
    is_due: bool,
) -> Value {
    json!({
        "id": question.id,
        "version": question.version,
        "title": question.title,
        "objective": question.objective,
        "conceptIds": question.concept_ids,
        "itemType": question.item_type,
        "prompt": question.prompt,
        "options": question.options,
        "sourceRefs": question.source_refs,
        "learnerState": state,
        "isDue": is_due,
        "whyDue": reason,
    })
}

fn attempt_response(event: Value, question: &QuestionRecord) -> Value {
    json!({
        "ok": true,
        "event": event,
        "feedback": {
            "correct": event.get("correct").and_then(Value::as_bool).unwrap_or(false),
            "correctOption": question.correct_option,
            "explanation": question.explanation,
            "confusable": question.confusable,
            "remediation": question.remediation,
            "sourceRefs": question.source_refs,
            "scheduledReviewAt": event.get("scheduled_review_at").cloned().unwrap_or(Value::Null),
            "nextReason": event.get("next_reason").cloned().unwrap_or(Value::Null),
            "errorClass": event.get("error_class").cloned().unwrap_or(Value::Null),
        }
    })
}

fn attempts_for<'a>(events: &'a [Value], question_id: &str) -> Vec<&'a Value> {
    let mut attempts: Vec<&Value> = events
        .iter()
        .filter(|event| {
            event.get("event_type").and_then(Value::as_str) == Some("question_attempt")
                && event.get("question_id").and_then(Value::as_str) == Some(question_id)
        })
        .collect();
    attempts.sort_by(|left, right| {
        left.get("occurred_at")
            .and_then(Value::as_str)
            .cmp(&right.get("occurred_at").and_then(Value::as_str))
    });
    attempts
}

fn index_attempts<'a>(events: &'a [Value]) -> BTreeMap<String, Vec<&'a Value>> {
    let mut index: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for event in events {
        if event.get("event_type").and_then(Value::as_str) != Some("question_attempt") {
            continue;
        }
        let Some(question_id) = event.get("question_id").and_then(Value::as_str) else {
            continue;
        };
        index
            .entry(question_id.to_string())
            .or_default()
            .push(event);
    }
    for attempts in index.values_mut() {
        attempts.sort_by(|left, right| {
            left.get("occurred_at")
                .and_then(Value::as_str)
                .cmp(&right.get("occurred_at").and_then(Value::as_str))
        });
    }
    index
}

fn evidence_state(attempts: &[&Value], now: DateTime<Utc>) -> EvidenceState {
    let event_refs = attempts
        .iter()
        .filter_map(|event| {
            event
                .get("event_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>();
    let next_review = attempts
        .last()
        .and_then(|event| event.get("scheduled_review_at"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let last_success = attempts
        .iter()
        .rev()
        .find(|event| event.get("correct").and_then(Value::as_bool) == Some(true))
        .and_then(|event| event.get("occurred_at"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let last_failure = attempts
        .iter()
        .rev()
        .find(|event| event.get("correct").and_then(Value::as_bool) == Some(false))
        .and_then(|event| event.get("occurred_at"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let modalities = attempts
        .iter()
        .filter(|event| event.get("correct").and_then(Value::as_bool) == Some(true))
        .filter_map(|event| event.get("evidence_modality").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let unresolved = attempts
        .iter()
        .filter(|event| event.get("correct").and_then(Value::as_bool) == Some(false))
        .flat_map(|event| {
            event
                .get("misconception_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let (state, summary) = match attempts.last() {
        None => ("unseen", "No recorded evidence yet."),
        Some(latest) if latest.get("correct").and_then(Value::as_bool) != Some(true) => {
            ("fragile", "The latest attempt exposed an unresolved miss.")
        }
        Some(latest) => {
            let correct_attempts: Vec<&Value> = attempts
                .iter()
                .copied()
                .filter(|event| event.get("correct").and_then(Value::as_bool) == Some(true))
                .collect();
            let delayed = correct_attempts
                .first()
                .and_then(|first| parse_event_time(first))
                .zip(
                    correct_attempts
                        .last()
                        .and_then(|last| parse_event_time(last)),
                )
                .map(|(first, last)| last - first >= ChronoDuration::days(1))
                .unwrap_or(false);
            let validated = correct_attempts.len() >= 2 && delayed && modalities.len() >= 2;
            if validated {
                let stale = parse_event_time(latest)
                    .map(|timestamp| now - timestamp > ChronoDuration::days(14))
                    .unwrap_or(false);
                if stale {
                    ("stale", "Previously validated evidence is due for refresh.")
                } else {
                    (
                        "validated",
                        "Delayed success exists across more than one modality.",
                    )
                }
            } else if latest
                .get("confidence_before")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                <= 1
            {
                (
                    "exposed",
                    "Correct recognition exists, but confidence is low.",
                )
            } else {
                (
                    "developing",
                    "A correct attempt exists; delayed or alternate-modality evidence is still needed.",
                )
            }
        }
    };

    EvidenceState {
        state: state.to_string(),
        summary: summary.to_string(),
        evidence_count: attempts.len(),
        modalities,
        unresolved_misconceptions: unresolved,
        last_success,
        last_failure,
        next_review,
        event_refs,
    }
}

fn schedule_review(
    correct: bool,
    confidence: u8,
    prior_correct_count: usize,
    now: DateTime<Utc>,
) -> DateTime<Utc> {
    let days = if !correct || confidence <= 1 {
        1
    } else if prior_correct_count > 0 {
        7
    } else {
        3
    };
    now + ChronoDuration::days(days)
}

fn next_reason(correct: bool, confidence: u8, prior_correct_count: usize) -> &'static str {
    if !correct && confidence >= 2 {
        "A high-confidence miss signals a likely misconception. Repair the mechanism now and retrieve it again tomorrow."
    } else if !correct {
        "This concept is not yet reliable. Teach the mechanism, discriminate the nearest confusable, and retrieve it tomorrow."
    } else if confidence <= 1 {
        "The answer was correct but uncertain, so retrieval tomorrow will test whether the mechanism is stable."
    } else if prior_correct_count > 0 {
        "Repeated confident success earns a seven-day interval; another modality is still required for validation."
    } else {
        "One confident success earns a three-day interval. Delayed retrieval will determine whether it holds."
    }
}

fn parse_timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

fn parse_event_time(event: &Value) -> Option<DateTime<Utc>> {
    event
        .get("occurred_at")
        .and_then(Value::as_str)
        .and_then(parse_timestamp)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    fn fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!("llm-wiki-study-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join("study")).unwrap();
        let question = json!({
            "id": "sy0701-2.4-ntlm-relay-001",
            "version": 1,
            "title": "NTLM credential relay",
            "objective": { "id": "2.4", "label": "Analyze indicators of malicious activity" },
            "conceptIds": ["ntlm-credential-relay"],
            "itemType": "scenario",
            "prompt": "Which control most directly reduces SMB relay?",
            "options": [
                { "id": "A", "text": "Require SMB signing" },
                { "id": "B", "text": "Increase password length" }
            ],
            "correctOption": "A",
            "explanation": "Required SMB signing rejects the unsigned relayed session.",
            "confusable": { "title": "Replay", "explanation": "Replay reuses stored material." },
            "remediation": [{ "label": "Draw the flow", "kind": "teach_back" }],
            "sourceRefs": [{ "id": "official", "label": "Objectives", "locator": "2.4" }]
        });
        fs::write(root.join(QUESTIONS_PATH), format!("{question}\n")).unwrap();
        root
    }

    fn attempt_body(response: &str, response_ms: u64) -> String {
        json!({
            "questionId": "sy0701-2.4-ntlm-relay-001",
            "questionVersion": 1,
            "selectedResponse": response,
            "confidenceBefore": 3,
            "rationaleText": "The server must reject unsigned sessions.",
            "evidenceModality": "scenario",
            "responseMs": response_ms,
            "sessionId": "test-session"
        })
        .to_string()
    }

    #[test]
    fn next_question_never_exposes_the_answer_or_explanation() {
        let root = fixture();
        let response = next_questions(&root, 1).unwrap();
        let serialized = response.to_string();
        assert!(!serialized.contains("correctOption"));
        assert!(!serialized.contains("Required SMB signing rejects"));
        assert!(serialized.contains("sourceRefs"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn idempotent_attempt_appends_exactly_once() {
        let root = fixture();
        let first = record_attempt(&root, &attempt_body("A", 2_000), "attempt-1").unwrap();
        let second = record_attempt(&root, &attempt_body("A", 2_000), "attempt-1").unwrap();
        assert_eq!(first["event"]["event_id"], second["event"]["event_id"]);
        let raw = fs::read_to_string(root.join(EVENTS_PATH)).unwrap();
        assert_eq!(raw.lines().count(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_lines_are_quarantined_without_losing_valid_evidence() {
        let root = fixture();
        record_attempt(&root, &attempt_body("B", 3_000), "attempt-1").unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(root.join(EVENTS_PATH))
            .unwrap();
        writeln!(file, "{{bad json").unwrap();

        let state = learner_state(&root, Some("ntlm-credential-relay")).unwrap();
        assert_eq!(state["supportingEventCount"], 1);
        let quarantine = fs::read_to_string(root.join(QUARANTINE_PATH)).unwrap();
        assert!(quarantine.contains("bad json"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn oversized_jsonl_lines_fail_before_unbounded_parsing() {
        let root = fixture();
        fs::write(
            root.join(QUESTIONS_PATH),
            vec![b'x'; MAX_JSONL_LINE_BYTES + 1],
        )
        .unwrap();
        let error = next_questions(&root, 1).unwrap_err();
        assert!(error.message.contains("64 KiB safety limit"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn oversized_event_log_fails_with_a_bounded_capacity_error() {
        let root = fixture();
        fs::create_dir_all(root.join(".llm-wiki/learning/events")).unwrap();
        let file = File::create(root.join(EVENTS_PATH)).unwrap();
        file.set_len(MAX_EVENT_LOG_BYTES + 1).unwrap();
        let error = learner_state(&root, None).unwrap_err();
        assert_eq!(error.status, 507);
        assert!(error.message.contains("64 MiB safety limit"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_appends_remain_one_json_object_per_line() {
        let root = Arc::new(fixture());
        let handles: Vec<_> = (0..8)
            .map(|index| {
                let root = Arc::clone(&root);
                thread::spawn(move || {
                    record_attempt(
                        root.as_path(),
                        &attempt_body(if index % 2 == 0 { "A" } else { "B" }, 1_000 + index),
                        &format!("attempt-{index}"),
                    )
                    .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let raw = fs::read_to_string(root.join(EVENTS_PATH)).unwrap();
        assert_eq!(raw.lines().count(), 8);
        assert!(raw
            .lines()
            .all(|line| serde_json::from_str::<Value>(line).is_ok()));
        let _ = fs::remove_dir_all(root.as_path());
    }

    #[cfg(unix)]
    #[test]
    fn question_symlink_escape_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = fixture();
        let external = std::env::temp_dir().join(format!("questions-{}.jsonl", Uuid::new_v4()));
        fs::copy(root.join(QUESTIONS_PATH), &external).unwrap();
        fs::remove_file(root.join(QUESTIONS_PATH)).unwrap();
        symlink(&external, root.join(QUESTIONS_PATH)).unwrap();
        assert!(next_questions(&root, 1).is_err());
        let _ = fs::remove_file(external);
        let _ = fs::remove_dir_all(root);
    }
}
