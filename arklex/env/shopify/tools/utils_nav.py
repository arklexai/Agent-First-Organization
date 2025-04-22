PAGEINFO_SLOTS = [
    {
        "name": "limit",
        "type": "string",
        "description": "Maximum number of entries to show.",
        "prompt": "",
        "required": False
    },
    {
        "name": "navigate",
        "type": "string",
        "description": "navigate relative to previous view. 'next' to search after previous view, 'prev' to search before the previous view. 'stay' or None to remain.'",
        "prompt": "",
        "required": False
    },
    {
        "name": "pageInfo",
        "type": "string",
        "description": "The previous pageInfo object, such as \"{'endCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9', 'hasNextPage': True, 'hasPreviousPage': False, 'startCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9'}\"",
        "prompt": "",
        "required": False
    },
]
    
PAGEINFO_OUTPUTS = [
    {
        "name": "pageInfo",
        "type": "string",
        "description": "Current pageInfo object, such as  \"{'endCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9', 'hasNextPage': True, 'hasPreviousPage': False, 'startCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9'}\""
    }
]

NAVIGATE_WITH_NO_CURSOR = "error: cannot navigate without reference cursor"
NO_NEXT_PAGE = "error: no more pages after"
NO_PREV_PAGE = "error: no more pages before"
    
def cursorify(kwargs):
    limit = int(kwargs.get('limit')) if kwargs.get('limit') else 3
    navigate = kwargs.get('navigate') if kwargs.get('navigate') else 'stay'
    pageInfo = kwargs.get('pageInfo')
    
    nav_param = f'first: {limit}'
    if navigate and navigate != 'stay':
        if not pageInfo:
            return NAVIGATE_WITH_NO_CURSOR, False
        
        if navigate == 'next':
            if not pageInfo['hasNextPage']:
                return NO_NEXT_PAGE, False
            nav_param = f"first: {limit}, after: \"{pageInfo['endCursor']}\""
            
        elif navigate == 'prev': 
            if not pageInfo['hasPreviousPage']:
                return NO_PREV_PAGE, False
            nav_param = f"last: {limit}, before: \"{pageInfo['startCursor']}\""
    
    return nav_param, True