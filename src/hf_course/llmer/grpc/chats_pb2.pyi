from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StartSessionRequest(_message.Message):
    __slots__ = ("user", "mode")
    USER_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    user: str
    mode: str
    def __init__(
        self, user: _Optional[str] = ..., mode: _Optional[str] = ...
    ) -> None: ...

class StartSessionResponse(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class Question(_message.Message):
    __slots__ = ("session_id", "system_prompt", "question_prompt")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    QUESTION_PROMPT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    system_prompt: str
    question_prompt: str
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        system_prompt: _Optional[str] = ...,
        question_prompt: _Optional[str] = ...,
    ) -> None: ...

class Answer(_message.Message):
    __slots__ = ("session_id", "text")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    text: str
    def __init__(
        self, session_id: _Optional[str] = ..., text: _Optional[str] = ...
    ) -> None: ...
