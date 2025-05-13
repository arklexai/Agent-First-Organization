from arklex.env.types import Resource, ResourceType
RESOURCES = [
    ############## WORKER ##############
    Resource(
        id="MessageWorker",
        name="MessageWorker",
        path="message_worker.py",
        type=ResourceType.WORKER,
        fixed_args={},
    ),
    Resource(
        id="FaissRAGWorker",
        name="FaissRAGWorker",
        path="faiss_rag_worker.py",
        type=ResourceType.WORKER,
        fixed_args={},

    ),

    ############## TOOL ##############
    Resource(
        id="ddbe6adc-cd0e-40bc-8a95-91cb69ed807b",
        name="find_contact_by_email",
        path="hubspot/find_contact_by_email.py",
        type=ResourceType.TOOL,
        fixed_args={"access_token": "<access_token>"},
    ),
    Resource(
        id="aa8dd20d-fda7-475b-91ce-8c5fc356a2b7",
        name="create_ticket",
        path="hubspot/create_ticket.py",
        type=ResourceType.TOOL,
        fixed_args={"access_token": "<access_token>"},
    ),
    Resource(
        id="8a6784c2-a130-4eb4-9924-4f4c58f4bf9d",
        name="check_available",
        path="hubspot/check_available.py",
        type=ResourceType.TOOL,
        fixed_args={"access_token": "<access_token>"},
    ),
    Resource(
        id="e86daf21-41a3-40b2-9695-3ed59be46cc4",
        name="create_meeting",
        path="hubspot/create_meeting.py",
        type=ResourceType.TOOL,
        fixed_args={"access_token": "<access_token>"},
    ),
    Resource(
        id="11860b97-dfcf-4f1d-9e44-8767c50fd371",
        name="find_owner_id_by_contact_id",
        path="hubspot/find_owner_id_by_contact_id.py",
        type=ResourceType.TOOL,
        fixed_args={"access_token": "<access_token>"},
    ),

]
