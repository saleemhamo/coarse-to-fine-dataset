import os

def main():
    # Print Hello
    print("Hello")

    # Print the current directory
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")

    # Print the directory content inside the mounted claim
    mounted_claim_directory = "/mnt/data"
    if os.path.exists(mounted_claim_directory):
        print(f"Contents of {mounted_claim_directory}:")
        for root, dirs, files in os.walk(mounted_claim_directory):
            for name in dirs:
                print(os.path.join(root, name))
            for name in files:
                print(os.path.join(root, name))
    else:
        print(f"Directory {mounted_claim_directory} does not exist.")

if __name__ == "__main__":
    main()
