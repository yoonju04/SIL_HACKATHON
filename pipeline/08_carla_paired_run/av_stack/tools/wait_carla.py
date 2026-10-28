import sys, time
import carla
c = carla.Client("localhost", 2000)
c.set_timeout(5.0)
t0 = time.time()
while True:
    try:
        print("server", c.get_server_version(), "after", round(time.time() - t0), "s", flush=True)
        break
    except Exception as e:
        if time.time() - t0 > 300:
            print("server not ready:", e); sys.exit(1)
        time.sleep(3)
