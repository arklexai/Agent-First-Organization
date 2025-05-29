import asyncio
from typing import Any, Dict, List, Tuple

from arklex.memory.core import ShortTermMemory
from arklex.utils.graph_state import BotConfig, LLMConfig, MessageState, ResourceRecord
from arklex.utils.model_config import MODEL

# class ResourceRecord(BaseModel):
#     info: Dict
#     intent: str = Field(default="")
#     input: List = Field(default_factory=list)
#     output: str = Field(default="")
#     steps: List = Field(default_factory=list)
#     personalized_intent: str = Field(default="")


# Case 1: General product list → Agent lists multiple categories → User then specifies one
record_case1 = ResourceRecord(
    info={"attribute": {"task": "List available products", "direct": False}},
    intent="User is browsing product categories",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – 3D unicorn design, ages 2‑12, adjustable buckle. Inventory: 547\n"
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets, one‑size fits all. Inventory: 349\n"
        "• Winter Flannel Blanket Solid Color Plaid Coral Blanket Fleece Bedspread (id=gid://shopify/Product/7949279526959) – Microfiber flannel, heated, anti‑pilling. Inventory: 0"
    ),
    steps=[
        {
            "context_generate": (
                "We have several products in our store. Are you looking for something particular? Let us know so we can find you the best match!"
            )
        }
    ],
)

# Case 1 Follow‑up: User specifies "Denim Apron with 5 Pockets"
record_case1_followup = ResourceRecord(
    info={"attribute": {"task": "Fetch product details", "direct": True}},
    intent="User wants product details",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets, one‑size fits all. Price: $19.99, Inventory: 349"
    ),
    steps=[
        {
            "context_generate": (
                "We have a versatile apron that could match your needs for various activities like BBQ, gardening, or painting. Are you interested in a particular color, like navy blue or black? Would you prefer an apron with specific features or price range?"
            )
        }
    ],
)


# Case 2: User asks about aprons → Agent lists aprons → User asks "Does it have pockets?"
record_case2_initial = ResourceRecord(
    info={"attribute": {"task": "List available products", "direct": False}},
    intent="User wants information about products",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": (
                "We found some aprons that might interest you. Are you looking for a specific color or need a certain number of pockets for your activities? Let us know how you'll be using the apron to better assist you in finding the perfect match!"
            )
        }
    ],
)

record_case2_followup = ResourceRecord(
    info={"attribute": {"task": "Fetch product details", "direct": True}},
    intent="User has inquiries related to product",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": "Yes—we have aprons with pockets for tools and accessories."
        }
    ],
)


# Case 3: Recommend aprons → then user switches to hats
record_case3_aprons = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": (
                "We have some aprons that might interest you! Do you have a color preference between navy blue, black, or blue? Also, are you looking for an apron with more pockets for functionality or fewer pockets for simplicity?"
            )
        }
    ],
)

record_case3_hats = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wanst recommendation",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – unicorn design, ages 2‑12. Inventory: 547\n"
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "We have some colorful and fun hats available. Are you looking for a particular color or style? Would you prefer an adjustable buckle for a better fit?"
            )
        }
    ],
)


# Case 4: Recommend hats → then navy blue hats follow‑up
record_case4_hats = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – 3D unicorn design. Inventory: 547\n"
        "• Green Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281296431) – reflective strip. Inventory: 55\n"
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – reflective strip, breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "We have a variety of baseball hats: unicorn, green no‑logo, and metallic navy blue. Interested in a specific color?"
            )
        }
    ],
)

record_case4_navy = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – reflective strip, breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "Sure—here's the navy blue baseball hat we have in stock."
            )
        }
    ],
)


# Collect all records in a list for testing
sample_records = [
    record_case1,
    record_case1_followup,
    record_case2_initial,
    record_case2_followup,
    record_case3_aprons,
    record_case3_hats,
    record_case4_hats,
    record_case4_navy,
]


# Shopify-style grouped records function
def get_shopify_records() -> List[List[ResourceRecord]]:
    """
    Returns sample ResourceRecord groups simulating Shopify ecommerce assistant turns.
    """
    return [
        record_case1,
        record_case1_followup,
        record_case2_initial,
        record_case2_followup,
        record_case3_aprons,
        record_case3_hats,
        record_case4_hats,
        record_case4_navy,
    ]


# Test configuration
TEST_CONFIG: Dict[str, Any] = {
    "model": MODEL,
    "role": "test_assistant",
    "user_objective": "Test the short term memory functionality",
    "builder_objective": "Ensure proper memory retrieval and intent matching",
    "intro": "This is a test bot for short term memory",
    "bot_id": "test_bot",
    "version": "1.0",
    "language": "EN",
    "bot_type": "test",
}


def init_test_state() -> MessageState:
    # Initialize LLMConfig from test config
    llm_config: LLMConfig = LLMConfig(**TEST_CONFIG.get("model", MODEL))

    # Create BotConfig
    bot_config: BotConfig = BotConfig(
        bot_id=TEST_CONFIG.get("bot_id", "test_bot"),
        version=TEST_CONFIG.get("version", "1.0"),
        language=TEST_CONFIG.get("language", "EN"),
        bot_type=TEST_CONFIG.get("bot_type", "test"),
        llm_config=llm_config,
    )

    # Create MessageState
    sys_instruct: str = (
        "You are a "
        + TEST_CONFIG["role"]
        + ". "
        + TEST_CONFIG["user_objective"]
        + TEST_CONFIG["builder_objective"]
        + TEST_CONFIG["intro"]
    )

    return MessageState(sys_instruct=sys_instruct, bot_config=bot_config)


def run_test_case(
    case_name: str,
    description: str,
    trajectory: List[List[ResourceRecord]],
    chat_history: str,
    query: str,
    expected_intent: bool,
    expected_record: bool,
    llm_config: LLMConfig,
) -> Tuple[bool, bool]:
    """Helper function to run a test case with standardized format.

    Args:
        case_name: Name of the test case
        description: Description of what the test case is testing
        trajectory: List of records for the test case
        chat_history: Chat history string
        query: Query string to test
        expected_intent: Expected found_intent value
        expected_record: Expected found_record value
        llm_config: LLM configuration

    Returns:
        Tuple of (found_intent, found_record) results
    """
    print(f"\n=== Case {case_name}: {description} ===")

    # Create STM instance
    stm = ShortTermMemory(trajectory, chat_history, llm_config=llm_config)
    asyncio.run(stm.personalize())

    # Run retrievals
    found_record, records = stm.retrieve_records(query)
    found_intent, intent = stm.retrieve_intent(query)

    # Print results
    print("Results:")
    print(f"- found_intent: {found_intent} (Expected: {expected_intent})")
    print(f"- found_record: {found_record} (Expected: {expected_record})")

    if found_record:
        print("\nFound Records Info:")
        for record in records:
            print(f"Record Intent: {record.intent}")
            print(f"Record Personalized Intent: {record.personalized_intent}")
            print(f"Record Output (first 100 tokens): {record.output[:100]}")

    return found_intent, found_record


def test_shopify_intent() -> None:
    # Initialize test state
    state = init_test_state()

    # Define test case labels
    test_case_labels = {
        "case1": {
            "found_intent_label": True,
            "found_record_label": True,
            "description": "Product Selection Flow",
        },
        "case2": {
            "found_intent_label": True,
            "found_record_label": True,
            "description": "Follow-up Questions",
        },
        "case3": {
            "found_intent_label": False,
            "found_record_label": False,
            "description": "Different Product Recommendations",
        },
        "case4": {
            "found_intent_label": True,
            "found_record_label": True,
            "description": "Same Product, Different Attributes",
        },
    }

    # Initialize LLM config
    llm_config = LLMConfig(**TEST_CONFIG.get("model", MODEL))
    bot_config = BotConfig(
        bot_id="test_bot",
        version="1.0",
        language="EN",
        bot_type="test",
        llm_config=llm_config,
    )

    # Test Case 1: Product Selection Flow
    trajectory_case1 = [[record_case1]]
    chat_history_case1 = """assistant: Hello! How can I help you today?
user: What products do you have?
assistant: We have several products in our store. Are you looking for something particular?"""
    found_intent_case1, found_record_case1 = run_test_case(
        "1",
        test_case_labels["case1"]["description"],
        trajectory_case1,
        chat_history_case1,
        "Show me the denim apron with 5 pockets",
        test_case_labels["case1"]["found_intent_label"],
        test_case_labels["case1"]["found_record_label"],
        bot_config.llm_config,
    )

    # Test Case 2: Follow-up Questions
    trajectory_case2 = [[record_case2_initial]]
    chat_history_case2 = """assistant: Hello! How can I help you today?
user: Do you have any aprons?
assistant: Yes, we have several aprons available."""
    found_intent_case2, found_record_case2 = run_test_case(
        "2",
        test_case_labels["case2"]["description"],
        trajectory_case2,
        chat_history_case2,
        "Does it have pockets?",
        test_case_labels["case2"]["found_intent_label"],
        test_case_labels["case2"]["found_record_label"],
        bot_config.llm_config,
    )

    # Test Case 3: Different Product Recommendations
    trajectory_case3 = [[record_case3_aprons]]
    chat_history_case3 = """assistant: Hello! How can I help you today?
user: I need an apron.
assistant: I can help you find the perfect apron."""
    found_intent_case3, found_record_case3 = run_test_case(
        "3",
        test_case_labels["case3"]["description"],
        trajectory_case3,
        chat_history_case3,
        "I want bedframes",
        test_case_labels["case3"]["found_intent_label"],
        test_case_labels["case3"]["found_record_label"],
        bot_config.llm_config,
    )

    # Test Case 4: Same Product, Different Attributes
    trajectory_case4 = [[record_case4_hats]]
    chat_history_case4 = """assistant: Hello! How can I help you today?
user: Show me hats.
assistant: Here are our hat collections."""
    found_intent_case4, found_record_case4 = run_test_case(
        "4",
        test_case_labels["case4"]["description"],
        trajectory_case4,
        chat_history_case4,
        "Do you have navy blue hats?",
        test_case_labels["case4"]["found_intent_label"],
        test_case_labels["case4"]["found_record_label"],
        bot_config.llm_config,
    )

    # Summary of Test Results
    print("\n=== Test Summary ===")
    actual_results = {
        "case1": {
            "found_intent": found_intent_case1,
        },
        "case2": {
            "found_intent": found_intent_case2,
        },
        "case3": {
            "found_intent": found_intent_case3,
        },
        "case4": {
            "found_intent": found_intent_case4,
        },
    }

    for case, labels in test_case_labels.items():
        actual = actual_results[case]
        intent_status = (
            "PASS" if actual["found_intent"] == labels["found_intent_label"] else "FAIL"
        )

        print(f"\n{case} ({labels['description']}):")
        print(
            f"  Intent: {intent_status} (Expected: {labels['found_intent_label']}, Got: {actual['found_intent']})"
        )


if __name__ == "__main__":
    test_shopify_intent()
