import time
import random
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

def print_colored_text(text, delay=0.1):
    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
    for char in text:
        color = random.choice(colors)
        print(color + char, end='', flush=True)
        time.sleep(delay)
    print(Style.RESET_ALL)

def main():
    # Clear screen
    print("\033[H\033[J", end="")
    
    # Print fancy border
    print(Fore.YELLOW + "=" * 50)
    
    # Print title with animation
    print_colored_text("Welcome to the Enhanced Hello World!", 0.05)
    
    # Print fancy border
    print(Fore.YELLOW + "=" * 50)
    
    # Print main message with different colors
    messages = [
        (Fore.RED + "Hello" + Style.RESET_ALL),
        (Fore.GREEN + "Beautiful" + Style.RESET_ALL),
        (Fore.BLUE + "World!" + Style.RESET_ALL)
    ]
    
    print("\n" + " ".join(messages))
    
    # Print decorative elements
    print("\n" + Fore.CYAN + "✨" + Style.RESET_ALL + " Made with Python " + Fore.CYAN + "✨" + Style.RESET_ALL)
    
    # Print bottom border
    print("\n" + Fore.YELLOW + "=" * 50)

if __name__ == "__main__":
    main()
