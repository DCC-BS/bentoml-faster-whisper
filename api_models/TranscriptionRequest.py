from pathlib import Path
from typing import Annotated, Any

from annotated_types import Ge, Le
from bentoml.exceptions import InvalidArgument
from pydantic import AliasChoices, BeforeValidator, ConfigDict, Field

from api_models.decode_params import DecodeParams
from api_models.enums import Language
from api_models.input_models import TimestampGranularities
from api_models.output_models import logger
from config import faster_whisper_config


def _process_empty_language(language: None | Language | str | bytes) -> Language | None:
    if isinstance(language, bytes):
        try:
            language = language.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                f"Cannot decode bytes language: {language}. Using default: {faster_whisper_config.default_language}"
            )
            return Language(value=faster_whisper_config.default_language)

    if language == "" or language is None:
        return None

    if isinstance(language, Language):
        return language

    try:
        return Language(value=language)
    except ValueError:
        logger.warning(f"Invalid language: {language}. Using default: {faster_whisper_config.default_language}")
        return Language(value=faster_whisper_config.default_language)


ValidatedLanguage = Annotated[Language | None, BeforeValidator(_process_empty_language)]


def _process_language_candidates(value: Any) -> Any:
    """Accept a comma-separated string (or bytes) besides a plain list; empty means None.

    Unlike ``language``, an invalid code raises instead of silently falling back to the
    default language: the caller is explicitly constraining detection, and a typo must
    not quietly change which languages are considered."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not value:
        return None
    return value


ValidatedLanguageCandidates = Annotated[list[Language] | None, BeforeValidator(_process_language_candidates)]


class TranscriptionRequest(DecodeParams):
    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptionRequest":
        try:
            return TranscriptionRequest.model_validate(d)
        except TypeError as e:
            raise InvalidArgument(str(e))

    file: Path = Field(description="The path to the audio file to be transcribed.")
    language: ValidatedLanguage | None = Field(
        default=faster_whisper_config.default_language,
        description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
        "not set, the language will be detected in the first 30 secondss of audio.",
    )
    language_candidates: ValidatedLanguageCandidates = Field(
        default=None,
        validation_alias=AliasChoices("language_candidates", "language_candidates[]"),
        description="Languages the audio may contain, as language codes (list or comma-separated string, "
        'e.g. "de,fr"). Only used when `language` is not set and `diarization` is enabled: per-region '
        "language detection is then restricted to these candidates. If not set, the candidate set is "
        "derived from the audio itself.",
    )
    timestamp_granularities: TimestampGranularities = Field(
        default=faster_whisper_config.default_timestamp_granularities,
        validation_alias=AliasChoices("timestamp_granularities", "timestamp_granularities[]"),
        description="The timestamp granularities to populate for this transcription. response_format must be "
        "set verbose_json to use timestamp granularities.",
    )
    diarization: bool = Field(
        default=faster_whisper_config.diarization,
        description="If True, the model will attempt to separate speakers in the audio. The pyannote "
        "speech turns also replace the Silero VAD: Whisper only decodes the detected speech regions.",
    )
    diarization_speaker_count: Annotated[int, Ge(1), Le(6)] | None = Field(
        default=None,
        description="The number of speakers to separate in the audio. This argument is only used if diarization is True.",
    )
    progress_id: str | None = Field(
        default=None,
        description="A unique identifier for reporting the progress of a task.",
    )
