import cv2
import numpy as np



def text_poping( frame, frame_id ,  text, start_frame, color,duration=30 ):

        t = (frame_id - start_frame) / duration
        t = np.clip(t, 0, 1)

        # Bounce effect
        scale = 1.2 + 0.5*np.sin(np.pi*t)

        alpha = 1.0
        if t > 0.75:
            alpha = 1 - (t-0.75)/0.25

        overlay = frame.copy()

        text = text

        (w,h),_ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            4
        )

        x = frame.shape[1]//2 - w//2
        y = frame.shape[0]//2 

        cv2.putText(
            overlay,
            text,
            (x,y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            4,
            cv2.LINE_AA
        )

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1-alpha,
            0,
            frame
        )
        return frame 



def flash_screen( frame, frame_id , color, start_frame, duration=12):

        t = (frame_id-start_frame)/duration

        alpha = 0.35*(1-t)

        overlay = np.full_like(frame, color)

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1-alpha,
            0,
            frame
        )
        return frame 