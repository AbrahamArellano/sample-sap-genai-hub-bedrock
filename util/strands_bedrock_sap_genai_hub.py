"""SAP GenAI Hub model provider.

- Docs: https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core#aws-bedrock
- SDK Reference: https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/gen_ai_hub.html
"""

import json
import logging
import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    cast,
    TypeVar,
    Type,
    Callable,
)

from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from strands.types.content import ContentBlock, Messages
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.models import Model
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

# Import SAP GenAI Hub SDK
try:
    from gen_ai_hub.proxy.native.amazon.clients import Session
except ImportError:
    raise ImportError(
        "SAP GenAI Hub SDK is not installed. Please install it with: "
        "pip install 'generative-ai-hub-sdk[all]'"
    )

logger = logging.getLogger(__name__)

DEFAULT_SAP_GENAI_HUB_MODEL_ID = "amazon--nova-lite"

# Common error messages for context window overflow
CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
]


class SAPGenAIHubModel(Model):
    """SAP GenAI Hub model provider implementation.

    This implementation handles SAP GenAI Hub-specific features such as:
    - Tool configuration for function calling
    - Streaming responses
    - Context window overflow detection
    - Support for different model types (Nova, Claude, Titan)
    """

    class SAPGenAIHubConfig(TypedDict, total=False):
        """Configuration options for SAP GenAI Hub models.

        Attributes:
            additional_args: Any additional arguments to include in the request
            max_tokens: Maximum number of tokens to generate in the response
            model_id: The SAP GenAI Hub model ID (e.g., "amazon--nova-lite", "anthropic--claude-3-sonnet")
            stop_sequences: List of sequences that will stop generation when encountered
            streaming: Flag to enable/disable streaming. Defaults to True.
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity via nucleus sampling (alternative to temperature)
        """

        additional_args: Optional[Dict[str, Any]]
        max_tokens: Optional[int]
        model_id: str
        stop_sequences: Optional[List[str]]
        streaming: Optional[bool]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(
        self,
        **model_config: Unpack[SAPGenAIHubConfig],
    ):
        """Initialize provider instance.

        Args:
            **model_config: Configuration options for the SAP GenAI Hub model.
        """
        self.config = SAPGenAIHubModel.SAPGenAIHubConfig(
            model_id=DEFAULT_SAP_GENAI_HUB_MODEL_ID
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        # Initialize the SAP GenAI Hub client not BOTO3
        self.client = Session().client(model_name=self.config["model_id"])

    @override
    def update_config(self, **model_config: Unpack[SAPGenAIHubConfig]) -> None:  # type: ignore
        """Update the SAP GenAI Hub Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> SAPGenAIHubConfig:
        """Get the current SAP GenAI Hub Model configuration.

        Returns:
            The SAP GenAI Hub model configuration.
        """
        return self.config

    def _is_nova_model(self) -> bool:
        """Check if the current model is an Amazon Nova model.

        Returns:
            True if the model is an Amazon Nova model, False otherwise.
        """
        nova_models = ["amazon--nova-pro", "amazon--nova-micro", "amazon--nova-lite"]
        return self.config["model_id"] in nova_models

    def _is_claude_model(self) -> bool:
        """Check if the current model is an Anthropic Claude model.

        Returns:
            True if the model is an Anthropic Claude model, False otherwise.
        """
        return self.config["model_id"].startswith("anthropic--claude")

    def _is_titan_text_model(self) -> bool:
        """Check if the current model is an Amazon Titan Text model.

        Returns:
            True if the model is an Amazon Titan Text model, False otherwise.
        """
        titan_text_models = ["amazon--titan-text-lite", "amazon--titan-text-express"]
        return self.config["model_id"] in titan_text_models

    def _is_titan_embed_model(self) -> bool:
        """Check if the current model is an Amazon Titan Embedding model.

        Returns:
            True if the model is an Amazon Titan Embedding model, False otherwise.
        """
        return self.config["model_id"] == "amazon--titan-embed-text"

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a request for the SAP GenAI Hub model.
        Taken from documentation here: https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core#concept_ynz_mgh_tzb__section_kx4_qg4_mbc

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A formatted request for the SAP GenAI Hub model.
        """
        # Format request based on model type
        if self._is_nova_model():
            return self._format_nova_request(messages, tool_specs, system_prompt)
        elif self._is_claude_model():
            return self._format_claude_request(messages, tool_specs, system_prompt)
        elif self._is_titan_text_model():
            return self._format_titan_text_request(messages)
        elif self._is_titan_embed_model():
            return self._format_titan_embed_request(messages)
        else:
            raise ValueError(f"Unsupported model: {self.config['model_id']}")

    def _format_nova_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a request for Amazon Nova models.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A formatted request for Amazon Nova models.
        """
        request = {
            "messages": messages,
            "inferenceConfig": {
                key: value
                for key, value in [
                    ("maxTokens", self.config.get("max_tokens")),
                    ("temperature", self.config.get("temperature")),
                    ("topP", self.config.get("top_p")),
                    ("stopSequences", self.config.get("stop_sequences")),
                ]
                if value is not None
            },
        }

        # Add system prompt if provided
        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        # Add tool specs if provided
        if tool_specs:
            request["toolConfig"] = {
                "tools": [{"toolSpec": tool_spec} for tool_spec in tool_specs],
                "toolChoice": {"auto": {}},
            }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    def _format_claude_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a request for Anthropic Claude models.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A formatted request for Anthropic Claude models.
        """
        # For Claude models, we'll use the same format as Nova models
        # since we're using the converse API for both
        request = {
            "messages": messages,
            "inferenceConfig": {
                key: value
                for key, value in [
                    ("maxTokens", self.config.get("max_tokens")),
                    ("temperature", self.config.get("temperature")),
                    ("topP", self.config.get("top_p")),
                    ("stopSequences", self.config.get("stop_sequences")),
                ]
                if value is not None
            },
        }

        # Add system prompt if provided
        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        # Add tool specs if provided
        if tool_specs:
            request["toolConfig"] = {
                "tools": [{"toolSpec": tool_spec} for tool_spec in tool_specs],
                "toolChoice": {"auto": {}},
            }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    def _format_titan_text_request(self, messages: Messages) -> Dict[str, Any]:
        """Format a request for Amazon Titan Text models.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            A formatted request for Amazon Titan Text models.
        """
        # Extract the text from the last user message
        input_text = ""
        for message in reversed(messages):
            if message["role"] == "user" and "content" in message:
                content_blocks = message["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if "text" in block:
                            input_text = block["text"]
                            break
                if input_text:
                    break

        # Create the request body
        request_body = {
            "inputText": input_text,
            "textGenerationConfig": {
                key: value
                for key, value in [
                    ("maxTokenCount", self.config.get("max_tokens")),
                    ("temperature", self.config.get("temperature")),
                    ("topP", self.config.get("top_p")),
                    ("stopSequences", self.config.get("stop_sequences")),
                ]
                if value is not None
            },
        }

        # Format the request according to the invoke_model API requirements
        request = {
            "body": json.dumps(request_body),
            "contentType": "application/json",
            "accept": "application/json",
        }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    def _format_titan_embed_request(self, messages: Messages) -> Dict[str, Any]:
        """Format a request for Amazon Titan Embedding models.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            A formatted request for Amazon Titan Embedding models.
        """
        # Extract the text from the last user message
        input_text = ""
        for message in reversed(messages):
            if message["role"] == "user" and "content" in message:
                content_blocks = message["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if "text" in block:
                            input_text = block["text"]
                            break
                if input_text:
                    break

        request = {
            "inputText": input_text,
        }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    @override
    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format the SAP GenAI Hub response events into standardized message chunks.

        Args:
            event: A response event from the SAP GenAI Hub model.

        Returns:
            The formatted chunk.
        """
        return cast(StreamEvent, event)

    def _convert_non_streaming_to_streaming(
        self, response: Dict[str, Any]
    ) -> Iterable[StreamEvent]:
        """Convert a non-streaming response to the streaming format.

        Args:
            response: The non-streaming response from the SAP GenAI Hub model.

        Returns:
            An iterable of response events in the streaming format.
        """
        if self._is_nova_model() or self._is_claude_model():
            # Nova and Claude models have a similar response format when using converse API
            # Yield messageStart event
            yield {"messageStart": {"role": response["output"]["message"]["role"]}}

            # Process content blocks
            for content in response["output"]["message"]["content"]:
                # Yield contentBlockStart event if needed
                if "toolUse" in content:
                    yield {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "toolUseId": content["toolUse"]["toolUseId"],
                                    "name": content["toolUse"]["name"],
                                }
                            },
                        }
                    }

                    # For tool use, we need to yield the input as a delta
                    input_value = json.dumps(content["toolUse"]["input"])

                    yield {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": input_value}}
                        }
                    }
                elif "text" in content:
                    # Then yield the text as a delta
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": content["text"]},
                        }
                    }

                # Yield contentBlockStop event
                yield {"contentBlockStop": {}}

            # Yield messageStop event
            yield {
                "messageStop": {
                    "stopReason": response.get("stopReason", "stop"),
                }
            }

            # Yield metadata event
            if "usage" in response or "metrics" in response:
                metadata: StreamEvent = {"metadata": {}}
                if "usage" in response:
                    metadata["metadata"]["usage"] = response["usage"]
                if "metrics" in response:
                    metadata["metadata"]["metrics"] = response["metrics"]
                yield metadata

        elif self._is_titan_text_model():
            # Titan Text models have a different response format
            # Yield messageStart event
            yield {"messageStart": {"role": "assistant"}}

            # Yield content block for text
            if "results" in response and len(response["results"]) > 0:
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": response["results"][0]["outputText"]},
                    }
                }

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

            # Yield messageStop event
            yield {
                "messageStop": {
                    "stopReason": "stop",
                }
            }

        elif self._is_titan_embed_model():
            # Titan Embedding models have a different response format
            # Yield messageStart event
            yield {"messageStart": {"role": "assistant"}}

            # Yield content block for embedding
            if "embedding" in response:
                yield {
                    "contentBlockDelta": {
                        "delta": {
                            "text": f"Embedding generated with {len(response['embedding'])} dimensions"
                        },
                    }
                }

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

            # Yield messageStop event
            yield {
                "messageStop": {
                    "stopReason": "stop",
                }
            }

    @override
    def stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        """Send the request to the SAP GenAI Hub model and get the response.

        This method calls the appropriate API based on the model type.

        Args:
            request: The formatted request to send to the SAP GenAI Hub model

        Returns:
            An iterable of response events from the SAP GenAI Hub model

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """
        streaming = self.config.get("streaming", True)

        try:
            if self._is_nova_model() or self._is_claude_model():
                if streaming:
                    # TODO: Implement streaming for Nova/Claude models when SAP GenAI Hub supports it
                    # For now, use non-streaming and convert to streaming format
                    response = self.client.converse(**request)
                    yield from self._convert_non_streaming_to_streaming(response)
                else:
                    response = self.client.converse(**request)
                    yield from self._convert_non_streaming_to_streaming(response)

            elif self._is_titan_text_model() or self._is_titan_embed_model():
                # Titan models don't support streaming
                response = self.client.invoke_model(**request)
                yield from self._convert_non_streaming_to_streaming(response)

        except Exception as e:
            error_message = str(e)

            # Handle throttling error
            if "ThrottlingException" in error_message:
                raise ModelThrottledException(error_message) from e

            # Handle context window overflow
            if any(
                overflow_message in error_message
                for overflow_message in CONTEXT_WINDOW_OVERFLOW_MESSAGES
            ):
                logger.warning("SAP GenAI Hub threw context window overflow error")
                raise ContextWindowOverflowException(e) from e

            # Otherwise raise the error
            raise e

    def structured_output(self, schema_type: Type, **kwargs: Any) -> Any:
        """Generate a structured output based on the provided schema.

        Args:
            schema_type: The Pydantic model class that defines the output schema.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            An instance of the provided schema_type.
        """
        from strands.event_loop.streaming import process_stream
        from strands.handlers.callback_handler import PrintingCallbackHandler
        from strands.tools import convert_pydantic_to_tool_spec

        # Create a tool spec from the schema
        tool_spec = convert_pydantic_to_tool_spec(schema_type)

        # Create a system prompt that instructs the model to generate a structured output
        system_prompt = f"""You are a helpful assistant that generates structured data according to a specific schema.
        
        The user will provide you with a request, and you must respond with a valid instance of the following schema:
        
        {schema_type.schema_json(indent=2)}
        
        Your response should be a valid JSON object that conforms to this schema.
        """

        # Create a message with the user's request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"Generate a structured output according to the provided schema based on this request: {kwargs.get('prompt', '')}"
                    }
                ],
            }
        ]

        # Format the request
        request = self.format_request(
            messages=messages,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
        )

        # Stream the response
        chunks = self.stream(request)

        # Process the stream to extract the structured output
        result = None
        for event in process_stream(chunks, messages):
            if "callback" in event and "delta" in event["callback"]:
                delta = event["callback"]["delta"]
                if "toolUse" in delta and "input" in delta["toolUse"]:
                    # Parse the tool use input as JSON
                    try:
                        input_json = json.loads(delta["toolUse"]["input"])
                        # Create an instance of the schema type from the JSON
                        result = schema_type(**input_json)
                    except Exception as e:
                        logger.error(f"Error parsing structured output: {e}")
                        raise ValueError(f"Failed to parse structured output: {e}")

        if result is None:
            raise ValueError("Failed to generate structured output")

        return result
