import threading
import unittest
from deployment_control import Assistance, answer_request, device_lock, deployment_active

class DeploymentControlTests(unittest.TestCase):
    def test_guidance_unblocks_waiting_worker(self):
        control=Assistance(); results=[]
        worker=threading.Thread(target=lambda:results.append(control.request({"missing":"Clock"})))
        worker.start()
        try:
            for _ in range(100):
                if control.pending: break
                threading.Event().wait(.01)
            self.assertIn("Provide more information",answer_request(control.token,""))
            self.assertEqual(results,[])
            answer_request(control.token,"Open Tools folder")
            worker.join(1)
            self.assertEqual(results,[{"skip":False,"info":"Open Tools folder"}])
        finally:
            control.close(); worker.join(1)

    def test_close_unblocks_worker_and_expires_token(self):
        control=Assistance(); results=[]
        worker=threading.Thread(target=lambda:results.append(control.request({"missing":"Clock"})))
        worker.start(); control.close(); worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(results[0]["skip"])
        self.assertIn("no longer active",answer_request(control.token,"info"))

    def test_skip_and_device_lock(self):
        control=Assistance(); results=[]
        worker=threading.Thread(target=lambda:results.append(control.request({"missing":"Clock"})))
        worker.start()
        try:
            for _ in range(100):
                if control.pending: break
                threading.Event().wait(.01)
            answer_request(control.token,skip=True); worker.join(1)
            self.assertEqual(results,[{"skip":True,"info":""}])
            lock=device_lock("test-device")
            self.assertIs(lock,device_lock("test-device"))
            lock.acquire()
            try: self.assertTrue(deployment_active())
            finally: lock.release()
        finally:
            control.close(); worker.join(1)
