package com.example.leafrecognizer;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.ShutterCallback;
import android.os.Build;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.MeasureSpec;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.RelativeLayout.LayoutParams;

public class CameraActivity extends Activity /*implements AdapterView.OnItemSelectedListener*/ {

	static final int REQUEST_IMAGE_CAPTURE = 1;
    private RelativeLayout mLayout;
	private Camera mCamera;
    private CameraPreview mPreview;
    private PictureCallback mPicture;
	private final static String TAG = "TouchFocus";
	private boolean listenerSet = false;
	public Paint paint;
	private DrawingView drawingView;
	private boolean drawingViewSet = false;
	private int counter;
	private String filename;
	private String filename2;
	//private int rectX;
	//private int rectY;
	//private ScheduledExecutorService myScheduledExecutorService;
	
	//private ImageView fotoButton;
    //private TextView textTimeLeft; // time left field

	//String path = Environment.getExternalStorageDirectory().getAbsolutePath();
	String path = Environment.getExternalStorageDirectory().getPath() + File.separator + "LeafRecog";
	private Context context;

	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);        
        setContentView(R.layout.camera_activity);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        context = this;
        
        mLayout = (RelativeLayout) findViewById(R.id.layout);
        //textTimeLeft=(TextView)findViewById(R.id.textTimeLeft); // make time left object
        // fotoButton = (ImageView) findViewById(R.id.photo_button);
		//fotoButton.setOnClickListener(captureListener);
		  // prompt = (TextView)findViewById(R.id.prompt);
        //mLayout.setOnTouchListener(captureListener);
		mPicture = getPictureCallback();

    }
    
    @Override
    protected void onResume() {
        super.onResume();
       
        createCameraPreview();

    }
    
	
	
	/**
	 * Called from PreviewSurfaceView to set touch focus.
	 * 
	 * @param - Rect - new area for auto focus
	 */
	public void doTouchFocus(final Rect tfocusRect) {
		
		try {
			counter=1;
			final List<Camera.Area> focusList = new ArrayList<Camera.Area>();
			Camera.Area focusArea = new Camera.Area(tfocusRect, 1000);
			focusList.add(focusArea);
		  
			Camera.Parameters para = mCamera.getParameters();
			para.setFocusAreas(focusList);
			para.setMeteringAreas(focusList);
			mCamera.setParameters(para);
			/*//Delay call autoFocus(myAutoFocusCallback)
			myScheduledExecutorService = Executors.newScheduledThreadPool(1);
			myScheduledExecutorService.schedule(new Runnable(){
			      public void run() {
			    	  mCamera.autoFocus(myAutoFocusCallback);
			        }
			      }, 500, TimeUnit.MILLISECONDS);*/
			mCamera.autoFocus(myAutoFocusCallback);
			Log.i(TAG, "TouchFocus");
			
		} catch (Exception e) {
			e.printStackTrace();
			Log.i(TAG, "Unable to autofocus");
		}

	}
	
	/**
	 * AutoFocus callback
	 */
	AutoFocusCallback myAutoFocusCallback = new AutoFocusCallback(){

		  @Override
		  public void onAutoFocus(boolean arg0, Camera arg1) {
		   if (arg0){
			   Camera.Parameters para = mCamera.getParameters();
			   para.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
               mCamera.setParameters(para);
			   mCamera.cancelAutoFocus();
			   mCamera.takePicture(null, null, mPicture);
			   }
		   
	    }
	};
	
	

	private PictureCallback getPictureCallback() {
		PictureCallback picture = new PictureCallback() {

			@Override
			public void onPictureTaken(byte[] data, Camera camera) {
				
				if(counter == 1){
				//make a new picture file
                    filename = savePicture(data);

                    if(filename == null) {
                        //ritorna al menu iniziale
                        gotoMainActivity();
                    }
                    else {
                        // Turn on flash light and take the next picture

                        // TODO questa riga era commentata !!!
                        mCamera.startPreview();

                        Camera.Parameters para = mCamera.getParameters();
                        para = mCamera.getParameters();
                        para.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
                        mCamera.setParameters(para);


                        counter--;
                        mCamera.takePicture(null, null, mPicture);

                    }
				}
				else {
					filename2 = savePicture(data);

					if(filename2 == null) {
						//ritorna al menu iniziale
						gotoMainActivity();
					}
					else {

                        Log.d("ERRORE", "Invio le 2 foto a ImageProcessActivity #########");

						startIPActivity(filename,filename2);
					}
                }

			}
		};

		return picture;
	}

    
	
    @Override
    protected void onPause() {
        super.onPause();
        mPreview.stop();
        mLayout.removeView(mPreview);
        mPreview = null;
    }
	
    private void createCameraPreview() {
        // Set the second argument by your choice.
        // Usually, 0 for back-facing camera, 1 for front-facing camera.
        // If the OS is pre-gingerbreak, this does not have any effect.
    	

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
            mCamera = Camera.open(0);
            
        } else {
            mCamera = Camera.open();
        }
    	
    	
        mPreview = new CameraPreview(this, mCamera, CameraPreview.LayoutMode.FitToParent, false);
        //dv = new DrawView(this);
        LayoutParams previewLayoutParams = new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT);
        mLayout.addView(mPreview, 0, previewLayoutParams);
        //mLayout.addView(dv);
        
        Rect rect = new Rect();
        mLayout.getDrawingRect(rect);

        mPreview.setPreviewSize(640, 480, rect.width(), rect.height());
        this.setListener(mPreview);
        drawingView = (DrawingView) findViewById(R.id.drawing_surface);
		this.setDrawingView(drawingView);


    }
    

    
    private String savePicture(byte[] data) {
    	File pictureFileDir = new File(path);
        if (!pictureFileDir.exists() && !pictureFileDir.mkdirs()) {
            Toast.makeText(context, "Can't create directory to save image.",
              Toast.LENGTH_LONG).show();
            return null;
        }

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyymmddhhmmss");
        String date = dateFormat.format(new Date());
        String photoFile = "Picture_" + date + ".jpg";
        String filenamez = pictureFileDir.getPath() + File.separator + photoFile;
        File pictureFile = new File(filenamez);

        try {
          FileOutputStream fos = new FileOutputStream(pictureFile);
          fos.write(data);
          fos.close();
          //Toast.makeText(context, "New Image saved:" + photoFile, Toast.LENGTH_LONG).show();
        } catch (Exception error) {
          Toast.makeText(context, "Image could not be saved.", Toast.LENGTH_LONG).show();
        }

        return filenamez;
    }
    
	@Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
    
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }
    
    public void gotoMainActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }

	public void startIPActivity(String filename,String filename2) {
        Intent intent = new Intent(this, ImageProcessActivity.class);
        intent.putExtra("photoPath", filename);
        intent.putExtra("photoPath2", filename2);
        intent.putExtra("stringa_prova", "Ecco la stringa di prova!");
        startActivity(intent);
    }
	
	
	@Override
	 public boolean onTouchEvent(MotionEvent event) {
	  if (!listenerSet) {
		  return false;
	  }
	  if(event.getAction() == MotionEvent.ACTION_DOWN){
		  float x = event.getX();
	      float y = event.getY();
	      
	      Rect touchRect = new Rect(
	    		(int)(x - 100), 
	  	        (int)(y - 100), 
	  	        (int)(x + 100), 
	  	        (int)(y + 100));
	      
	      final Rect targetFocusRect = new Rect(
					touchRect.left * 2000/mPreview.getWidth() - 1000,
				    touchRect.top * 2000/mPreview.getHeight() - 1000,
				    touchRect.right * 2000/mPreview.getWidth() - 1000,
				    touchRect.bottom * 2000/mPreview.getHeight() - 1000);
	      
	      this.doTouchFocus(targetFocusRect);
	      
	      if (drawingViewSet) {
	    	  drawingView.setHaveTouch(true, touchRect);
	    	  drawingView.invalidate();
	    	  
	    	  // Remove the square after some time
	    	  Handler handler = new Handler();
	    	  handler.postDelayed(new Runnable() {
				
				@Override
				public void run() {
					drawingView.setHaveTouch(false, new Rect(0, 0, 0, 0));
					drawingView.invalidate();
					
				}
			}, 1000);
	      }
	      
	  }
	  return false;
	 }
	
	 /**
	  * set CameraPreview instance for touch focus.
	  * @param camPreview - CameraPreview
	  */
	public void setListener(CameraPreview camPreview) {
		listenerSet = true;
	}

	/**
	 * set DrawingView instance for touch focus indication.
	 * @param dView
     */
	public void setDrawingView(DrawingView dView) {
		drawingView = dView;
		drawingViewSet = true;
	}
	
	
}
    