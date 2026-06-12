import register_user
import train_model
import detect_live
from utils import create_dirs

def main():
    create_dirs()
    while True:
        print("\n--- Face & Gender Detection System ---")
        print("1. Register User")
        print("2. Train Model")
        print("3. Live Detection")
        print("4. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1': register_user.register()
        elif choice == '2': train_model.train()
        elif choice == '3': detect_live.detect()
        elif choice == '4': break
        else: print("Invalid Choice!")

if __name__ == "__main__":
    main()