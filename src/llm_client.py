"""
Клиент LLM без внешних зависимостей (только stdlib).

Поддерживаются два провайдера:

1. provider: "yandex" — нативный API Yandex Foundation Models
   POST https://llm.api.cloud.yandex.net/foundationModels/v1/completion
   Переменные окружения:
       YC_API_KEY    — API-ключ сервисного аккаунта (заголовок Authorization: Api-Key ...)
       YC_IAM_TOKEN  — альтернатива API-ключу (заголовок Authorization: Bearer ...), живёт 12 часов
       YC_FOLDER_ID  — идентификатор каталога Yandex Cloud, подставляется в modelUri
       LLM_MODEL     — имя модели, например yandexgpt-5-lite/latest или yandexgpt-5-lite/rc

2. provider: "openai_compatible" — POST {base_url}/chat/completions
   Подходит для OpenAI, DeepSeek, OpenRouter, vLLM, LM Studio, Ollama
   (base_url = http://localhost:11434/v1, ключ не требуется), а также для
   OpenAI-совместимого эндпоинта Yandex Cloud.
   Переменные окружения:
       LLM_API_KEY     — ключ API
       LLM_BASE_URL    — базовый URL, переопределяет llm.base_url
       LLM_MODEL       — имя модели, переопределяет llm.model
       LLM_AUTH_SCHEME — схема авторизации: Api-Key или Bearer.
                         Если не задана, выбирается автоматически: для ключа Yandex Cloud
                         (AQVN...) — Api-Key, для IAM-токена (t1....) и всех остальных
                         провайдеров — Bearer.

3. provider: "none" — генерация отключена, выполняется только retrieval.

Ключи читаются ТОЛЬКО из переменных окружения / .env и никогда не попадают
в конфиги, логи и артефакты.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


def load_dotenv(path: str = ".env") -> None:
    """
    Минимальный загрузчик .env (чтобы не тянуть python-dotenv).
    Уже установленные переменные окружения не перезаписываются.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class LLMError(RuntimeError):
    pass


def _mask(secret: str) -> str:
    """Ключ в логах и артефактах показываем только хвостом."""
    if not secret:
        return "не задан"
    return f"…{secret[-4:]}"


class LLMClient:
    SUPPORTED = ('yandex', 'openai_compatible', 'none')

    def __init__(self, llm_config: Dict[str, Any]):
        load_dotenv()
        self.provider = str(llm_config.get('provider', 'openai_compatible')).lower()
        if self.provider not in self.SUPPORTED:
            raise LLMError(
                f"Неподдерживаемый llm.provider='{self.provider}'. "
                f"Доступно: {', '.join(self.SUPPORTED)}."
            )

        self.model = os.environ.get('LLM_MODEL') or llm_config.get('model', '')
        self.temperature = float(llm_config.get('temperature', 0.0))
        self.max_tokens = int(llm_config.get('max_tokens', 700))
        self.timeout = int(llm_config.get('timeout_seconds', 60))

        if self.provider == 'yandex':
            self._init_yandex(llm_config)
        elif self.provider == 'openai_compatible':
            self._init_openai(llm_config)

    # ---------- инициализация провайдеров ----------

    def _init_yandex(self, llm_config: Dict[str, Any]) -> None:
        # У нативного API свой путь .../foundationModels/v1/completion.
        # base_url из конфига рассчитан на OpenAI-совместимый слой (.../v1) и сюда не годится:
        # POST нативного payload на .../v1 возвращает 404. Поэтому берём внешнее значение,
        # только если это действительно эндпоинт foundationModels.
        candidate = (os.environ.get('LLM_BASE_URL') or llm_config.get('base_url') or '').rstrip('/')
        self.base_url = candidate if 'foundationmodels' in candidate.lower() else YANDEX_COMPLETION_URL
        self.folder_id = os.environ.get('YC_FOLDER_ID', '')
        self.api_key = os.environ.get('YC_API_KEY', '')
        self.iam_token = os.environ.get('YC_IAM_TOKEN', '')
        self.model = self.model or 'yandexgpt-5-lite/latest'

        if not self.folder_id:
            raise LLMError(
                "Не задан YC_FOLDER_ID. Это идентификатор каталога Yandex Cloud "
                "(консоль -> каталог -> поле ID вида b1g...). Пропишите его в .env."
            )
        if not (self.api_key or self.iam_token):
            raise LLMError(
                "Не задан ни YC_API_KEY, ни YC_IAM_TOKEN. Нужен API-ключ сервисного "
                "аккаунта с ролью ai.languageModels.user (или IAM-токен). Пропишите в .env."
            )

        # modelUri: gpt://<folder_id>/<model>. Если в LLM_MODEL уже полный URI — берём как есть.
        self.model_uri = self.model if self.model.startswith('gpt://') \
            else f"gpt://{self.folder_id}/{self.model}"

    def _default_auth_scheme(self) -> str:
        """
        Api-Key — для API-ключа Yandex Cloud, Bearer — для IAM-токена и прочих провайдеров.
        IAM-токен опознаётся по префиксу 't1.', API-ключ — по 'AQVN'.
        """
        if 'yandex' not in getattr(self, 'base_url', ''):
            return 'Bearer'
        if self.api_key.startswith('t1.'):
            return 'Bearer'
        return 'Api-Key'

    def _init_openai(self, llm_config: Dict[str, Any]) -> None:
        self.base_url = (os.environ.get('LLM_BASE_URL')
                         or llm_config.get('base_url', 'https://api.openai.com/v1')).rstrip('/')
        self.api_key = os.environ.get('LLM_API_KEY', '')
        # Yandex Cloud ждёт «Authorization: Api-Key <секрет>» для API-ключа сервисного
        # аккаунта и «Bearer <токен>» для IAM-токена. Жёсткий Bearer давал 401 на ключе
        # AQVN..., поэтому схема выбирается автоматически; LLM_AUTH_SCHEME переопределяет.
        self.auth_scheme = os.environ.get('LLM_AUTH_SCHEME') or self._default_auth_scheme()
        self.model = self.model or llm_config.get('model', 'gpt-4o-mini')
        self.folder_id = os.environ.get('YC_FOLDER_ID', '')

        # Yandex Cloud в OpenAI-совместимом режиме ждёт в поле model полный URI
        # вида gpt://<folder_id>/<модель>. Если указано короткое имя, достраиваем его
        # из YC_FOLDER_ID — иначе API вернёт 400/404 с невнятным текстом.
        if 'yandex' in self.base_url and not self.model.startswith('gpt://'):
            if not self.folder_id:
                raise LLMError(
                    "Для Yandex Cloud нужен YC_FOLDER_ID в .env либо полный идентификатор "
                    "модели в LLM_MODEL (например gpt://b1g.../yandexgpt-5-lite/latest)."
                )
            self.model = f"gpt://{self.folder_id}/{self.model}"

        if 'yandex' in self.base_url and not self.api_key:
            raise LLMError(
                "Не задан LLM_API_KEY. Нужен API-ключ сервисного аккаунта Yandex Cloud "
                "с ролью ai.languageModels.user."
            )

    # ---------- общее ----------

    @property
    def enabled(self) -> bool:
        return self.provider != 'none'

    def describe(self) -> str:
        if self.provider == 'none':
            return "LLM отключён (llm.provider: none) — выполняется только retrieval."
        if self.provider == 'yandex':
            auth = f"IAM {_mask(self.iam_token)}" if self.iam_token else f"Api-Key {_mask(self.api_key)}"
            return f"YandexGPT: {self.model_uri} ({auth})"
        key_state = _mask(self.api_key) if self.api_key else "без ключа (локальный эндпоинт)"
        return f"LLM: {self.model} @ {self.base_url} ({key_state})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.enabled:
            raise LLMError("LLM отключён в конфиге (llm.provider: none).")
        if self.provider == 'yandex':
            return self._complete_yandex(system_prompt, user_prompt)
        return self._complete_openai(system_prompt, user_prompt)

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            hint = ""
            is_yandex = 'yandex' in getattr(self, 'base_url', '')
            if exc.code == 400 and is_yandex:
                hint = (" Скорее всего в поле model ушло короткое имя без префикса "
                        "gpt://<folder_id>/. Проверьте YC_FOLDER_ID и LLM_MODEL.")
            elif exc.code == 401:
                hint = (" Не принята авторизация. Для API-ключа Yandex Cloud нужна схема "
                        "Api-Key, для IAM-токена — Bearer (переменная LLM_AUTH_SCHEME).")
            elif exc.code == 403:
                hint = (" Нет прав. Проверьте роль ai.languageModels.user на каталоге, "
                        "привязанный платёжный аккаунт и область действия (scope) API-ключа.")
            elif exc.code == 404:
                hint = (" Модель не найдена. Сверьте идентификатор с кнопкой «URI инстанса» "
                        "в карточке модели Yandex AI Studio (например yandexgpt-5-lite/latest).")
            elif exc.code == 429:
                hint = " Превышена квота каталога по запросам/токенам. Смотрите «Квоты» в консоли."
            raise LLMError(f"LLM вернул HTTP {exc.code}: {detail}.{hint}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Не удалось подключиться к {url}: {exc.reason}. Проверьте сеть и URL."
            ) from exc

    # ---------- YandexGPT ----------

    def _complete_yandex(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),  # API ожидает строку
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }
        headers = {"x-folder-id": self.folder_id}
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        else:
            headers["Authorization"] = f"Api-Key {self.api_key}"

        body = self._post(self.base_url, payload, headers)
        try:
            return body["result"]["alternatives"][0]["message"]["text"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(f"Неожиданный формат ответа YandexGPT: {json.dumps(body, ensure_ascii=False)[:500]}") from exc

    # ---------- OpenAI-совместимый ----------

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"{self.auth_scheme} {self.api_key}"

        body = self._post(f"{self.base_url}/chat/completions", payload, headers)
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(f"Неожиданный формат ответа LLM: {json.dumps(body, ensure_ascii=False)[:500]}") from exc


def create_client(llm_config: Dict[str, Any]):
    """
    Безопасная инициализация: возвращает (client, error).

    Если конфигурация неполна (нет YC_FOLDER_ID, не задан ключ, неизвестный провайдер),
    возвращается (None, текст ошибки) вместо traceback — пайплайн в этом случае
    отрабатывает retrieval и честно сообщает, почему нет генерации.
    """
    try:
        return LLMClient(llm_config), None
    except LLMError as exc:
        return None, str(exc)
