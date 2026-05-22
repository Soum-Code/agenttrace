import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

def main():
    print("=" * 60)
    print("STARTING ADVANCED FOCUS BROWSER AUTOMATION TEST FOR AGENTTRACE")
    print("=" * 60)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1280,1400")  # taller screen to capture complete dashboard and results
    
    print("[+] Initializing headless Chrome WebDriver...")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("[+] Navigating to Streamlit Dashboard at http://127.0.0.1:8501...")
        driver.get("http://127.0.0.1:8501")
        
        print("[+] Waiting 8 seconds for websocket handshake and page rendering...")
        time.sleep(8)
        
        print("[+] Finding text area input...")
        text_areas = driver.find_elements(By.TAG_NAME, "textarea")
        if not text_areas:
            print("[-] Error: Streamlit text area not found!")
            return
            
        target_text_area = text_areas[0]
        test_prompt = "Find the population of Tokyo and compare it with Delhi, then calculate the ratio."
        print(f"[+] Entering prompt: '{test_prompt}'")
        target_text_area.click()
        target_text_area.clear()
        target_text_area.send_keys(test_prompt)
        time.sleep(1)
        
        # Press Tab and Enter/Escape to force lose focus and bind state in Streamlit
        print("[+] Pressing TAB to lose focus and bind text area state...")
        target_text_area.send_keys(Keys.TAB)
        time.sleep(2)
        
        # Also click on header to ensure focus is completely lost from input
        try:
            header_el = driver.find_element(By.CLASS_NAME, "main-header")
            header_el.click()
            print("[+] Clicked header to ensure loss of focus.")
        except Exception:
            pass
        time.sleep(2)
        
        # Find all buttons
        print("[+] Looking for active buttons...")
        buttons = driver.find_elements(By.TAG_NAME, "button")
        analyze_btn = None
        for b in buttons:
            if "Analyze Trace" in b.text:
                analyze_btn = b
                break
                
        if not analyze_btn:
            print("[-] 'Analyze Trace' button text not found. Attempting class based lookup...")
            for b in buttons:
                if "primary" in b.get_attribute("class") or b.get_attribute("kind") == "primary":
                    analyze_btn = b
                    break
                    
        if not analyze_btn and buttons:
            analyze_btn = buttons[0]
            
        if not analyze_btn:
            print("[-] Error: No buttons found on page!")
            return
            
        print(f"[+] Clicking 'Analyze Trace' button (enabled={analyze_btn.is_enabled()})...")
        driver.execute_script("arguments[0].click();", analyze_btn)
        
        print("[+] Request submitted. Waiting 25 seconds for model execution and page refresh...")
        for i in range(5):
            time.sleep(5)
            print(f"    Waiting... {5 * (i+1)}s elapsed")
            
        # Scroll to the bottom of the page to ensure we capture all results
        print("[+] Scrolling down to capture full results...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Save screenshot
        screenshot_path = r"C:\Users\somna\.gemini\antigravity\brain\f00b83cd-bec4-459c-ab37-d93918452859\streamlit_screenshot.png"
        driver.save_screenshot(screenshot_path)
        print(f"[+] Screenshot captured successfully! Saved to: {screenshot_path}")
        print("[+] Testing complete. Closing WebDriver.")
        
    except Exception as e:
        print(f"[-] Automation error: {e}")
    finally:
        driver.quit()
        print("=" * 60)

if __name__ == "__main__":
    main()
