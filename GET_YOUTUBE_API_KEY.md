# How to Get Your YouTube API Key (5 Minutes)

## Step-by-Step Guide

### Step 1: Go to Google Cloud Console
Open this link in your browser:
👉 **https://console.cloud.google.com/**

Sign in with your Google account (any Gmail account works).

---

### Step 2: Create a New Project

1. Click the **project dropdown** at the top (says "Select a project")
2. Click **"NEW PROJECT"** button (top right)
3. Enter project name: `Lip-Reading-Dataset` (or any name you like)
4. Click **"CREATE"**
5. Wait ~30 seconds for project creation

**Screenshot location:** Top bar, left side

---

### Step 3: Enable YouTube Data API v3

**IMPORTANT:** Don't use the top search bar! Use one of these methods:

**Method 1 - Direct Link (EASIEST):**
1. Go to: https://console.cloud.google.com/apis/library/youtube.googleapis.com
2. Make sure your project "Lip-Reading-Dataset" is selected (check top bar)
3. Click the blue **"ENABLE"** button
4. Wait ~10 seconds

**Method 2 - Via APIs & Services:**
1. Click **"APIs & Services"** in the left sidebar
2. Click **"Library"**
3. In the search box (inside Library page), type: `YouTube Data API v3`
4. Click on **"YouTube Data API v3"** in the results
5. Click the blue **"ENABLE"** button

**Method 3 - Via Enable APIs Button:**
1. Go to **APIs & Services > Dashboard**
2. Click **"+ ENABLE APIS AND SERVICES"** (top button)
3. Search: `YouTube Data API v3`
4. Click on it and click **"ENABLE"**

**You'll see:** "API enabled" message or it will show as "Enabled"

---

### Step 4: Create API Key

1. Click **"Credentials"** in the left sidebar
   - Or go directly to: https://console.cloud.google.com/apis/credentials

2. Click **"CREATE CREDENTIALS"** button (top)

3. Select **"API key"** from the dropdown

4. **COPY YOUR API KEY!** It looks like:
   ```
   AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

5. **IMPORTANT:** Click **"RESTRICT KEY"** (recommended for security)

---

### Step 5: Restrict Your API Key (Security)

1. Under "API restrictions", select **"Restrict key"**

2. In the dropdown, find and check:
   - ✅ **YouTube Data API v3**

3. Click **"SAVE"**

**Why?** This prevents your key from being used for other Google APIs if it leaks.

---

### Step 6: Save Your API Key Securely

**DO NOT:**
- ❌ Share it publicly
- ❌ Commit it to GitHub
- ❌ Post it in Discord/Slack
- ❌ Email it to anyone

**DO:**
- ✅ Save it in a password manager
- ✅ Store it in a local file (already in .gitignore)
- ✅ Keep it private

---

## Quick Verification

Your API key should:
- Start with `AIza`
- Be about 39 characters long
- Contain letters, numbers, and some symbols

Example format: `AIzaSyABC123def456GHI789jkl012MNO345pqr`

---

## What's Next?

Once you have your API key, paste it in the terminal when I ask for it.

I'll then:
1. ✅ Save it securely (already protected by .gitignore)
2. ✅ Search for 800+ candidate videos
3. ✅ Check which ones are trainable
4. ✅ Download all trainable videos
5. ✅ Save everything in `youtube_raw_downloads/`

---

## Troubleshooting

### "I don't see CREATE CREDENTIALS button"
- Make sure you're in the correct project (check top bar)
- Make sure YouTube Data API v3 is enabled

### "API key doesn't work"
- Wait 1-2 minutes after creation (propagation time)
- Make sure you restricted it to YouTube Data API v3
- Try regenerating the key

### "Quota exceeded error"
- You've used your 10,000 daily units
- Wait until midnight Pacific Time for reset
- This is normal and FREE - no charges

---

## Security Reminder

✅ Your .gitignore is already updated to protect:
- `.youtube_api_key`
- `youtube_api_key.txt`
- `*api_key*.txt`
- `.env` files
- Any file with "secret" or "key" in the name

**You're safe!** Your API key won't be committed to GitHub.

---

## Ready?

Get your API key using the steps above, then paste it in the terminal!

Format: Just the key itself, like:
```
AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

I'll handle the rest! 🚀
