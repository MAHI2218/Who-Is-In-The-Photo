# Sample Data Directory

This folder holds ready-to-use test datasets for **"Who Is in the Photo?"**:

### 1. `known_faces/` (50 Reference Employee Portraits)
Contains 50 single-person portrait photos named in `[Firstname Lastname].jpg` format:
- **Male (25):** Aarav Patel, Alexander Wright, Carlos Mendez, David Kim, Ethan Davis, James Wilson, Liam Brown, Lucas Garcia, Marcus Johnson, Michael Chang, Noah Miller, Oliver Smith, Rahul Sharma, Robert Taylor, Ryan Thomas, Samuel Jackson, Sebastian Moore, Vikram Malhotra, William Anderson, Zack Martin, Brian Clark, Daniel Lewis, George Hall, Henry White, Jack Harris.
- **Female (25):** Amara Singh, Chloe Bennett, Elena Rostova, Emily Watson, Grace Hopper, Hannah Scott, Isabella Rodriguez, Jessica Alba, Katherine Johnson, Layla Hassan, Maya Lin, Mia Thompson, Nadia Petrova, Olivia Martin, Pooja Hegde, Priya Nair, Rachel Green, Sarah Connor, Sophia Chen, Tara Sharma, Victoria Adams, Zoe Saldana, Alice Walker, Clara Oswald, Diana Prince.

> All 50 images contain **exactly 1 face** and pass validation with 100% indexing rate.

---

### 2. `group_photos/` (Test Group Photos)
- **`office_team_meeting_1.jpg`:**
  - Contains **10 people**:
    - **6 Known Employees:** Aarav Patel, Alexander Wright, Carlos Mendez, Amara Singh, Chloe Bennett, Elena Rostova.
    - **4 Unknown Guests:** Unlisted individuals to test the **"Unknown"** badge and threshold logic.
- **`office_team_meeting_2.jpg`:**
  - Contains **10 people**:
    - **7 Known Employees:** David Kim, Ethan Davis, James Wilson, Emily Watson, Grace Hopper, Hannah Scott, Isabella Rodriguez.
    - **3 Unknown Guests:** Unlisted individuals.

---

### 3. Re-generating Data
To regenerate the dataset at any time, run:
```bash
python scripts/generate_sample_data.py
```
