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
        id="DataBaseWorker",
        name="DataBaseWorker",
        path="database_worker.py",
        type=ResourceType.WORKER,
        fixed_args={},

    ),

    ############## TOOL ##############
    Resource(
        id="ddbe6adc-cd0e-40bc-8a95-91cb69ed807b",
        name="search_show",
        path="booking_db/search_show.py",
        type=ResourceType.TOOL,
        fixed_args={},
    ),
    Resource(
        id="b9dbef8b-8219-4e0a-a50d-3b01614e5443",
        name="book_show",
        path="booking_db/book_show.py",
        type=ResourceType.TOOL,
        fixed_args={},
    ),
    Resource(
        id="2a2750cb-6226-4068-ba05-a4db83da3e16",
        name="check_booking",
        path="booking_db/check_booking.py",
        type=ResourceType.TOOL,
        fixed_args={},
    ),
    Resource(
        id="6b5d95df-1106-4044-a202-8fd38cef4d0e",
        name="cancel_booking",
        path="booking_db/cancel_booking.py",
        type=ResourceType.TOOL,
        fixed_args={},
    ),

]
