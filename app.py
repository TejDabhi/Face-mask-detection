import streamlit as st
import re

st.set_page_config(page_title="MAC Address Analyzer", layout="centered")

st.title("🔍 MAC Address Analyzer")
st.markdown("Validate and analyze Unicast, Multicast & Broadcast MAC Addresses")

# ------------------------------
# Sample OUI Database (Offline Example)
# ------------------------------
OUI_DATABASE = {
    "00:1A:2B": "Cisco Systems",
    "3C:5A:B4": "Dell Inc.",
    "F0:18:98": "Apple Inc.",
    "FC:FB:FB": "Google Inc."
}

# ------------------------------
# Functions
# ------------------------------

def normalize_mac(mac):
    mac = mac.upper().replace("-", ":").replace(".", "")
    if "." in mac:
        mac = mac.replace(".", "")
    if len(mac.replace(":", "")) == 12:
        mac = ":".join(mac.replace(":", "")[i:i+2] for i in range(0,12,2))
    return mac

def is_valid_mac(mac):
    pattern = r'^([0-9A-F]{2}:){5}[0-9A-F]{2}$'
    return re.match(pattern, mac)

def get_mac_type(mac):
    if mac == "FF:FF:FF:FF:FF:FF":
        return "Broadcast"

    first_byte = int(mac.split(":")[0], 16)

    if first_byte & 1:
        return "Multicast"
    else:
        return "Unicast"

def get_bits_info(mac):
    first_byte = int(mac.split(":")[0], 16)
    binary = format(first_byte, '08b')

    ig_bit = binary[-1]      # Individual/Group
    ul_bit = binary[-2]      # Universal/Local

    return binary, ig_bit, ul_bit

def get_vendor(mac):
    prefix = ":".join(mac.split(":")[0:3])
    return OUI_DATABASE.get(prefix, "Unknown Vendor")

# ------------------------------
# User Input
# ------------------------------

mac_input = st.text_area("Enter one or multiple MAC Addresses (one per line):")

if st.button("Analyze"):
    mac_list = mac_input.strip().split("\n")

    for mac in mac_list:
        mac = normalize_mac(mac.strip())

        st.markdown("---")
        st.subheader(f"MAC: {mac}")

        if not is_valid_mac(mac):
            st.error("❌ Invalid MAC Address Format")
            continue

        mac_type = get_mac_type(mac)
        binary, ig_bit, ul_bit = get_bits_info(mac)
        vendor = get_vendor(mac)

        # Display Results
        if mac_type == "Broadcast":
            st.success("📡 Type: BROADCAST")
        elif mac_type == "Multicast":
            st.warning("👥 Type: MULTICAST")
        else:
            st.info("👤 Type: UNICAST")

        st.write(f"**Vendor (OUI Lookup):** {vendor}")
        st.write(f"**First Octet (Binary):** {binary}")
        st.write(f"**I/G Bit (LSB):** {ig_bit} → {'Multicast' if ig_bit=='1' else 'Unicast'}")
        st.write(f"**U/L Bit:** {ul_bit} → {'Locally Administered' if ul_bit=='1' else 'Universally Administered'}")

        st.markdown("""
        🔎 **Explanation:**
        - I/G Bit (Least Significant Bit):  
          - 0 → Unicast  
          - 1 → Multicast  
        - U/L Bit:
          - 0 → Globally unique (assigned by vendor)
          - 1 → Locally modified MAC
        """)

st.markdown("---")
st.caption("Developed for CN Practical – MAC Address Validation Lab")
