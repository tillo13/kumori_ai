# bot_animation.py

import time

# ASCII art states for chatbot animation
states = [
    """
    +------+   
    |      |
    |      |
    +------+
     \____/
    """,
    """
    +------+   
    |      |
    |      O
    +------+
     \____/ 
    """,
    """
    +------+   
    |      |
    |      O
    +----OO+
     \____/ 
    """
]

def animate_chatbot():
    """Animate the chatbot's ASCII representation in an infinite loop."""
    while True:
        for state in states:
            print("\033[H\033[J")  # Clear the terminal screen
            print(state)
            time.sleep(0.5)

if __name__ == "__main__":
    animate_chatbot()