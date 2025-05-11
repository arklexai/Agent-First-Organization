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
        name="create_event",
        path="google/calendar/create_event.py",
        type=ResourceType.TOOL,
        fixed_args={"service_account_info": "<credential json content from service account app key>",
                    "delegated_user": "<service account>"}

    ),

]
