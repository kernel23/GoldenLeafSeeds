import streamlit_authenticator as stauth

# 1. Define the passwords you want to use
passwords = ['1THESS51618!l25','pass']

# 2. Hash them
hashed_passwords = stauth.Hasher(passwords).generate()

# 3. Print the hashes so you can copy them into your config file
for pw, h in zip(passwords, hashed_passwords):
    print(f"Password: {pw} -> Hash: {h}")