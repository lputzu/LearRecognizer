package com.example.leafrecognizer;

import java.io.IOException;
import java.util.List;

import android.app.Activity;
import android.content.res.Configuration;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.os.Build;
import android.os.Handler;
import android.util.Log;
import android.view.Display;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View.MeasureSpec;
import android.widget.RelativeLayout;
import android.widget.Toast;

/**
 * This class assumes the parent layout is RelativeLayout.LayoutParams.
 */
public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private static boolean DEBUGGING = true;
    private static final String LOG_TAG = "LeafRecognizer";
    private static final String CAMERA_PARAM_ORIENTATION = "orientation";
    private static final String CAMERA_PARAM_LANDSCAPE = "landscape";
    private Activity mActivity;
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private List<Size> mPreviewSizeList;
    private List<Size> mPictureSizeList;
    private Size mPreviewSize;
    private Size mPictureSize;
    private LayoutMode mLayoutMode;
    private int mCenterPosX = -1;
    private int mCenterPosY;
	private Paint textPaint = new Paint();
	private Paint circlePaint = new Paint();
	
    PreviewReadyCallback mPreviewReadyCallback = null;
    
    public static enum LayoutMode {
        FitToParent, // Scale to the size that no side is larger than the parent
        NoBlank // Scale to the size that no side is smaller than the parent
    };
    
    public interface PreviewReadyCallback {
        public void onPreviewReady();
    }
 
    /**
     * State flag: true when surface's layout size is set and surfaceChanged()
     * process has not been completed.
     */
    protected boolean mSurfaceConfiguring = false;
    
    
    

    public CameraPreview(Activity activity, Camera mCamera, LayoutMode mode, boolean addReversedSizes) {
        super(activity); // Always necessary
        this.mCamera = mCamera; 
        mActivity = activity;
        mLayoutMode = mode;
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        


        Parameters cameraParams = mCamera.getParameters();
        mPreviewSizeList = cameraParams.getSupportedPreviewSizes();
        mPictureSizeList = cameraParams.getSupportedPictureSizes();
        
        
        
        if (addReversedSizes) {
            List<Size> sizes = mPreviewSizeList;
            int length = sizes.size();
            for (int i = 0; i < length; i++) {
                Size size = sizes.get(i);
                Size revSize = mCamera.new Size(size.height, size.width);
                sizes.add(revSize);
            }
        }
        
        textPaint.setARGB(255, 200, 0, 0);
	    textPaint.setTextSize(60);
	    circlePaint.setColor(Color.YELLOW);
	    circlePaint.setStyle(Paint.Style.STROKE);
	    circlePaint.setStrokeWidth(5);
	    
	    
	    setWillNotDraw(false);
	    
	    
    }


    /**
     *
     * @param width is the width of the available area for this view
     * @param height is the height of the available area for this view
     * @param layoutWidth
     * @param layoutHeight
     */
    public void setPreviewSize(int width, int height, int layoutWidth, int layoutHeight) {
        mCamera.stopPreview();
        
        Parameters cameraParams = mCamera.getParameters();
        //boolean portrait = isPortrait();
        
        Size previewSize = determinePreviewSize(640, 480);
        
        //Toast.makeText(mActivity, "SetPreviewSize chiama DetPicSize con PreviewSize=(" + previewSize.width + ", " + previewSize.height + ")", Toast.LENGTH_LONG).show();

        
        Size pictureSize = determinePictureSize(previewSize);
        if (DEBUGGING) { Log.v(LOG_TAG, "Requested Preview Size - w: " + previewSize.width + ", h: " + previewSize.height); }
        mPreviewSize = previewSize;
        //Toast.makeText(mActivity, "Width: " + mPreviewSize.width + ", Height:" + mPreviewSize.height, Toast.LENGTH_LONG).show();
        mPictureSize = pictureSize;
        //Toast.makeText(mActivity, "Chiamata da SetPreviewSize con (" + layoutWidth + ", " + layoutHeight + ")", Toast.LENGTH_LONG).show();

        boolean layoutChanged = adjustSurfaceLayoutSize(previewSize, layoutWidth, layoutHeight);
        if (layoutChanged) {
            mSurfaceConfiguring = true;
            return;
        }

        configureCameraParameters(cameraParams);
        try {
            mCamera.startPreview();
        } catch (Exception e) {
            Toast.makeText(mActivity, "Failed to start preview: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
        mSurfaceConfiguring = false;
        
        
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        try {
            mCamera.setPreviewDisplay(mHolder);
        } catch (IOException e) {
            mCamera.release();
            mCamera = null;
        }
    }
    
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        mCamera.stopPreview();
        
        
        
        Parameters cameraParams = mCamera.getParameters();

        if (!mSurfaceConfiguring) {
            Size previewSize = determinePreviewSize(640, 480);
           // Toast.makeText(mActivity, "Surfacechanged chiama DetPicSize con PreviewSize=(" + previewSize.width + ", " + previewSize.height + ")", Toast.LENGTH_LONG).show();

            Size pictureSize = determinePictureSize(previewSize);
            if (DEBUGGING) { Log.v(LOG_TAG, "Desired Preview Size - w: " + width + ", h: " + height); }
            mPreviewSize = previewSize;
            mPictureSize = pictureSize;
            //Toast.makeText(mActivity, "Chiamata da surfaceChanged con (" + this.getWidth() + ", " + this.getHeight() + ")", Toast.LENGTH_LONG).show();

            mSurfaceConfiguring = adjustSurfaceLayoutSize(previewSize, width, height);
            if (mSurfaceConfiguring) {
                return;
            }
        }

        configureCameraParameters(cameraParams);
        mSurfaceConfiguring = false;

        try {
            mCamera.startPreview();
        } catch (Exception e) {
            Toast.makeText(mActivity, "Failed to start preview: " + e.getMessage(), Toast.LENGTH_LONG).show();
            Log.w(LOG_TAG, "Failed to start preview: " + e.getMessage());
        }
        
        
    }

    /**
     * @param reqWidth must be the value of the parameter passed in surfaceChanged
     * @param reqHeight must be the value of the parameter passed in surfaceChanged
     * @return Camera.Size object that is an element of the list returned from Camera.Parameters.getSupportedPreviewSizes.
     */
    protected Size determinePreviewSize(int reqWidth, int reqHeight) {
        // Meaning of width and height is switched for preview when portrait,
        // while it is the same as user's view for surface and metrics.
        // That is, width must always be larger than height for setPreviewSize.
//reqWidth;;  requested width in terms of camera hardware
 // reqHeight;; requested height in terms of camera hardware
    	

/*

        if (DEBUGGING) {
            Log.v(LOG_TAG, "Listing all supported preview sizes");
            for (Size size : mPreviewSizeList) {
                Log.v(LOG_TAG, "  w: " + size.width + ", h: " + size.height);
            }
            Log.v(LOG_TAG, "Listing all supported picture sizes");
            for (Size size : mPictureSizeList) {
                Log.v(LOG_TAG, "  w: " + size.width + ", h: " + size.height);
            }
        }
*/

        Size previewSize = mPreviewSizeList.get(0);

        for (Size size : mPreviewSizeList) {
            if(size.width <= reqWidth && size.height <= reqHeight)
            {
            	previewSize = size;
            	break;
            }
        }
        

        return previewSize;
        
        
//        // Adjust surface size with the closest aspect-ratio
//        float reqRatio = ((float) reqWidth) / reqHeight;
//        float curRatio, deltaRatio;
//        float deltaRatioMin = Float.MAX_VALUE;
//        Camera.Size retSize = null;
//        for (Camera.Size size : mPreviewSizeList) {
//            curRatio = ((float) size.width) / size.height;
//            deltaRatio = Math.abs(reqRatio - curRatio);
//            if (deltaRatio < deltaRatioMin) {
//                deltaRatioMin = deltaRatio;
//                retSize = size;
//            }
//        }
//
//        return retSize;
    }

    protected Size determinePictureSize(Size previewSize) {
        Size retSize = null;
        for (Size size : mPictureSizeList) {
            if (size.equals(previewSize)) {
            	//Toast.makeText(mActivity, "Picture (" + size.width + ", " + size.height + ") come previewSize", Toast.LENGTH_LONG).show();
                return size;
            }
        }
        
        if (DEBUGGING) { Log.v(LOG_TAG, "Same picture size not found."); }
        
        // if the preview size is not supported as a picture size
        float reqRatio = ((float) previewSize.width) / previewSize.height;
        float curRatio, deltaRatio;
        float deltaRatioMin = Float.MAX_VALUE;
        for (Size size : mPictureSizeList) {
            curRatio = ((float) size.width) / size.height;
            deltaRatio = Math.abs(reqRatio - curRatio);
            if (deltaRatio < deltaRatioMin) {
                deltaRatioMin = deltaRatio;
                retSize = size;
            }
        }
    	//Toast.makeText(mActivity, "Picture (" + retSize.width + ", " + retSize.height + ") in seguito calcolo", Toast.LENGTH_LONG).show();

        return retSize;
    }
    
    protected boolean adjustSurfaceLayoutSize(Size previewSize, int availableWidth, int availableHeight) {
        float tmpLayoutHeight, tmpLayoutWidth;
        tmpLayoutHeight = previewSize.height;
        tmpLayoutWidth = previewSize.width;
        
        //Toast.makeText(mActivity, "Availables=(" + availableWidth + ", " + availableHeight + ")", Toast.LENGTH_LONG).show();


        float factH, factW, fact;
        factH = availableHeight / tmpLayoutHeight;
        factW = availableWidth / tmpLayoutWidth;
        if (mLayoutMode == LayoutMode.FitToParent) {
            // Select smaller factor, because the surface cannot be set to the size larger than display metrics.
            if (factH < factW) {
                fact = factH;
            } else {
                fact = factW;
            }
        } else {
            if (factH < factW) {
                fact = factW;
            } else {
                fact = factH;
            }
        }

        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams)this.getLayoutParams();

        int layoutHeight = (int) (tmpLayoutHeight * fact);
        int layoutWidth = (int) (tmpLayoutWidth * fact);
        if (DEBUGGING) {
            Log.v(LOG_TAG, "Preview Layout Size - w: " + layoutWidth + ", h: " + layoutHeight);
            Log.v(LOG_TAG, "Scale factor: " + fact);
        }

        boolean layoutChanged;
        if ((layoutWidth != this.getWidth()) || (layoutHeight != this.getHeight())) {
            layoutParams.height = layoutHeight;
            layoutParams.width = layoutWidth;
            //Toast.makeText(mActivity, "Fact: " + fact + ", Layout=(" + layoutWidth + ", " + layoutHeight + ")", Toast.LENGTH_LONG).show();

            if (mCenterPosX >= 0) {
                layoutParams.topMargin = mCenterPosY - (layoutHeight / 2);
                layoutParams.leftMargin = mCenterPosX - (layoutWidth / 2);
            }
            this.setLayoutParams(layoutParams); // this will trigger another surfaceChanged invocation.
            layoutChanged = true;
        } else {
            layoutChanged = false;
        }

        
        return layoutChanged;
    }

    /**
     * @param x X coordinate of center position on the screen. Set to negative value to unset.
     * @param y Y coordinate of center position on the screen.
     */
    public void setCenterPosition(int x, int y) {
        mCenterPosX = x;
        mCenterPosY = y;
    }
    
    protected void configureCameraParameters(Parameters cameraParams) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.FROYO) { // for 2.1 and before
            cameraParams.set(CAMERA_PARAM_ORIENTATION, CAMERA_PARAM_LANDSCAPE);
            cameraParams.setFocusMode(Parameters.FOCUS_MODE_AUTO);
        } else { // for 2.2 and later
            int angle;
            Display display = mActivity.getWindowManager().getDefaultDisplay();
            switch (display.getRotation()) {
                case Surface.ROTATION_0: // This is display orientation
                    angle = 90; // This is camera orientation
                    break;
                case Surface.ROTATION_90:
                    angle = 0;
                    break;
                case Surface.ROTATION_180:
                    angle = 270;
                    break;
                case Surface.ROTATION_270:
                    angle = 180;
                    break;
                default:
                    angle = 90;
                    break;
            }
            Log.v(LOG_TAG, "angle: " + angle);
            mCamera.setDisplayOrientation(angle);
        }

        cameraParams.setPreviewSize(mPreviewSize.width, mPreviewSize.height);
        //Toast.makeText(mActivity, "PreviewSize=(" + mPreviewSize.width + ", " + mPreviewSize.height + ")", Toast.LENGTH_LONG).show();

        cameraParams.setPictureSize(mPictureSize.width, mPictureSize.height);
        if (DEBUGGING) {
            Log.v(LOG_TAG, "Preview Actual Size - w: " + mPreviewSize.width + ", h: " + mPreviewSize.height);
            Log.v(LOG_TAG, "Picture Actual Size - w: " + mPictureSize.width + ", h: " + mPictureSize.height);
        }
        cameraParams.setFocusMode(Parameters.FOCUS_MODE_AUTO);
        mCamera.setParameters(cameraParams);
    }

    @Override
	protected void onDraw(Canvas canvas){
		
	canvas.drawCircle(canvas.getWidth()/2, canvas.getHeight()/2, canvas.getWidth()/15, circlePaint);
       // Toast.makeText(mActivity, "Canvas: (" + this.getWidth() + ", " + this.getHeight() + ")", Toast.LENGTH_LONG).show();

	    
	}
	
	@Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        stop();
    }
    
	
    public void stop() {
        if (null == mCamera) {
            return;
        }
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
    }
	
	
	public void setOneShotPreviewCallback(PreviewCallback callback) {
		if (null == mCamera) {
	    	return;
		}
		mCamera.setOneShotPreviewCallback(callback);
	}
	    
	public void setPreviewCallback(PreviewCallback callback) {
		if (null == mCamera) {
	    	return;
		}
		mCamera.setPreviewCallback(callback);
	}
	
    public Size getPreviewSize() {
        return mPreviewSize;
    }
	
	public List<Size> getSupportedPreviewSizes() {
        return mPreviewSizeList;
    }
    
	
	public void setOnPreviewReady(PreviewReadyCallback cb) {
        mPreviewReadyCallback = cb;
    }	
	
	@Override
	 protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
		setMeasuredDimension(
			MeasureSpec.getSize(widthMeasureSpec),
			MeasureSpec.getSize(heightMeasureSpec));
	 }
}
