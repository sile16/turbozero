import os

import xml.etree.ElementTree as ET
import cairosvg
from PIL import Image
import jax

def render_pgx_2p(frames, p_ids, title, frame_dir, p1_label='Black', p2_label='White', duration=900):
    """really messy render function for rendering frames from a 2-player game
    from a PGX environment to a .gif"""
    
    try:
        # Ensure the target directory exists
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"Rendering {len(frames)} frames to {frame_dir}/{title}.gif")
        
        # Process all frames to create an animated GIF
        try:
            # Create temporary directory for frame images
            frame_temp_dir = f"{frame_dir}/temp"
            os.makedirs(frame_temp_dir, exist_ok=True)
            
            # Process each frame
            png_files = []
            for i, frame in enumerate(frames):
                # Get SVG content for this frame
                svg_content = frame.env_state.to_svg(color_theme='dark')
                
                # Save SVG file
                svg_path = f"{frame_temp_dir}/frame_{i:03d}.svg"
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                
                # Convert to PNG
                png_path = f"{frame_temp_dir}/frame_{i:03d}.png"
                cairosvg.svg2png(url=svg_path, write_to=png_path)
                png_files.append(png_path)
                
                print(f"Processed frame {i+1}/{len(frames)}")
                
                # Limit number of frames for performance if needed
                if i >= 59:  # Max 60 frames
                    break
            
            # Create animated GIF from all frames
            if png_files:
                frames_pil = [Image.open(png_file) for png_file in png_files]
                gif_path = f"{frame_dir}/{title}.gif"
                
                # Save as animated GIF
                if len(frames_pil) > 1:
                    frames_pil[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames_pil[1:],
                        duration=duration,
                        loop=0
                    )
                else:
                    # Single frame case
                    frames_pil[0].save(gif_path)
                    
                print(f"Created animated GIF with {len(frames_pil)} frames at {gif_path}")
                
                # Clean up temporary files
                for file_path in png_files:
                    os.remove(file_path)
                for i in range(len(png_files)):
                    svg_path = f"{frame_temp_dir}/frame_{i:03d}.svg"
                    if os.path.exists(svg_path):
                        os.remove(svg_path)
                
                # Remove temp directory
                try:
                    os.rmdir(frame_temp_dir)
                except:
                    pass
                
                return gif_path
            else:
                # No frames were processed successfully
                raise Exception("No frames were processed successfully")
                
        except Exception as e:
            print(f"Error creating animated GIF: {str(e)}")
            # Fallback to blank GIF
            blank_image = Image.new('RGB', (100, 100), color='black')
            gif_path = f"{frame_dir}/{title}.gif"
            blank_image.save(gif_path)
            print(f"Created a blank placeholder GIF at {gif_path}")
            return gif_path
            
    except Exception as e:
        import traceback
        print(f"Exception in render_pgx_2p: {e}")
        traceback.print_exc()
        
        # Create a blank GIF as a last resort
        try:
            blank_image = Image.new('RGB', (100, 100), color='black')
            gif_path = f"{frame_dir}/{title}.gif"
            blank_image.save(gif_path)
            print(f"Created a blank placeholder GIF after error at {gif_path}")
            return gif_path
        except:
            print("Failed to create even a blank GIF")
            return ""
