# Security Setup - IP Whitelisting

This guide shows you how to restrict access to your BTC DCA Dashboard to only your devices.

## How It Works

The app now includes IP-based access control. By default, it's **disabled** (allows all IPs). To enable it, you need to configure the `ALLOWED_IPS` environment variable on Render.

## Step-by-Step Setup

### 1. Find Your IP Address

Visit your deployed app at:
```
https://your-app.onrender.com/api/myip
```

This will show your current IP address in JSON format:
```json
{
  "your_ip": "123.456.789.0",
  "message": "Add this IP to ALLOWED_IPS environment variable on Render: 123.456.789.0"
}
```

### 2. Configure Render Environment Variable

1. Go to your [Render Dashboard](https://dashboard.render.com)
2. Select your web service (btc-dca-pwa)
3. Go to **Environment** tab
4. Click **Add Environment Variable**
5. Add:
   - **Key**: `ALLOWED_IPS`
   - **Value**: Your IP address(es), comma-separated

**Examples:**

Single IP:
```
ALLOWED_IPS=123.456.789.0
```

Multiple IPs (home, office, mobile):
```
ALLOWED_IPS=123.456.789.0,98.765.432.1,45.678.901.2
```

Disable IP filtering (allow all):
```
ALLOWED_IPS=all
```

### 3. Save and Redeploy

1. Click **Save Changes**
2. Render will automatically redeploy your app
3. Wait for deployment to complete (~2-3 minutes)

### 4. Test Access

- **From allowed IP**: You should see the dashboard normally
- **From other IP**: You'll get a `403 Forbidden` error

## Managing Multiple Devices

### Home Network
Your home IP usually stays the same, but it can change if your ISP reassigns it. Check `/api/myip` periodically.

### Mobile Devices
Mobile IPs change frequently when switching between WiFi and cellular. Options:
- Add your mobile carrier's IP when needed
- Use VPN with static IP
- Keep one "trusted" network for access

### Dynamic IPs
If your IP changes often, you have options:
1. Update the `ALLOWED_IPS` environment variable on Render
2. Use a VPN service with static IP
3. Use HTTP Basic Auth instead (see below)

## Alternative: HTTP Basic Authentication

If you prefer username/password instead of IP whitelisting, let me know and I can implement that instead.

## Testing Locally

When running locally (`python dca_dashboard.py`), IP filtering is controlled by the `ALLOWED_IPS` environment variable:

```bash
# Allow all (default)
python dca_dashboard.py

# Restrict to localhost only
ALLOWED_IPS=127.0.0.1 python dca_dashboard.py

# Restrict to specific IPs
ALLOWED_IPS=192.168.1.100,192.168.1.101 python dca_dashboard.py
```

## Security Notes

- IP whitelisting is **not perfect** security, but it's a good basic layer
- For stronger security, consider adding HTTP Basic Auth as well
- Always use HTTPS (Render provides this automatically)
- Don't share your Render app URL publicly

## Troubleshooting

**Problem**: Can't access the app after setting ALLOWED_IPS

**Solutions**:
1. Check your current IP at https://whatismyipaddress.com
2. Update `ALLOWED_IPS` on Render with the correct IP
3. Temporarily set `ALLOWED_IPS=all` to regain access
4. Check Render logs for "403 Forbidden" errors

**Problem**: IP keeps changing

**Solution**: Use HTTP Basic Auth instead of IP whitelisting (ask me to implement this)
